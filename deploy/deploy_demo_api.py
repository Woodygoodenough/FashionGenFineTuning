#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "deploy" / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "demo_api.json"
BUILD_DIR = STATE_DIR / "demo_api_build"
ZIP_PATH = STATE_DIR / "demo_api_lambda.zip"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = True) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def aws(cmd: list[str], *, region: str) -> dict | str:
    out = run(["aws", "--region", region, *cmd])
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def ensure_role(role_name: str, region: str) -> str:
    try:
        role = aws(["iam", "get-role", "--role-name", role_name, "--output", "json"], region=region)["Role"]
        return role["Arn"]
    except subprocess.CalledProcessError:
        assume_role = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        role = aws(
            [
                "iam",
                "create-role",
                "--role-name",
                role_name,
                "--assume-role-policy-document",
                json.dumps(assume_role),
                "--output",
                "json",
            ],
            region=region,
        )["Role"]
        run(
            [
                "aws",
                "iam",
                "attach-role-policy",
                "--role-name",
                role_name,
                "--policy-arn",
                "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            ]
        )
        time.sleep(10)
        return role["Arn"]


def build_zip() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    requirements = ROOT / "deploy" / "api_requirements.txt"
    if requirements.read_text().strip():
        run(
            [
                "python3",
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements),
                "-t",
                str(BUILD_DIR),
            ],
            capture=False,
        )

    pkg_dir = BUILD_DIR / "app"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "python" / "app" / "demo_search.py", pkg_dir / "demo_search.py")
    shutil.copy2(ROOT / "python" / "app" / "demo_api_lambda.py", pkg_dir / "demo_api_lambda.py")
    (pkg_dir / "__init__.py").write_text("")

    catalog_src = ROOT / "nextjs" / "public" / "demo" / "catalog.json"
    shutil.copy2(catalog_src, BUILD_DIR / "catalog.json")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in BUILD_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(BUILD_DIR))


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def ensure_function(region: str, role_arn: str) -> dict:
    state = load_state()
    function_name = "fashiongen-demo-api"
    try:
        info = aws(["lambda", "get-function", "--function-name", function_name, "--output", "json"], region=region)
        run(
            [
                "aws",
                "--region",
                region,
                "lambda",
                "update-function-code",
                "--function-name",
                function_name,
                "--zip-file",
                f"fileb://{ZIP_PATH}",
            ],
            capture=False,
        )
    except subprocess.CalledProcessError:
        info = aws(
            [
                "lambda",
                "create-function",
                "--function-name",
                function_name,
                "--runtime",
                "python3.12",
                "--role",
                role_arn,
                "--handler",
                "app.demo_api_lambda.handler",
                "--timeout",
                "30",
                "--memory-size",
                "1024",
                "--zip-file",
                f"fileb://{ZIP_PATH}",
                "--output",
                "json",
            ],
            region=region,
        )

    for _ in range(30):
        conf = aws(["lambda", "get-function-configuration", "--function-name", function_name, "--output", "json"], region=region)
        if conf["State"] == "Active" and conf["LastUpdateStatus"] in {"Successful", "InProgress"}:
            break
        time.sleep(2)

    try:
        url_cfg = aws(["lambda", "get-function-url-config", "--function-name", function_name, "--output", "json"], region=region)
        run(
            [
                "aws",
                "--region",
                region,
                "lambda",
                "update-function-url-config",
                "--function-name",
                function_name,
                "--auth-type",
                "NONE",
                "--cors",
                json.dumps(
                    {
                        "AllowOrigins": ["*"],
                        "AllowMethods": ["GET"],
                        "AllowHeaders": ["*"],
                    }
                ),
            ],
            capture=False,
        )
        url_cfg = aws(["lambda", "get-function-url-config", "--function-name", function_name, "--output", "json"], region=region)
    except subprocess.CalledProcessError:
        url_cfg = aws(
            [
                "lambda",
                "create-function-url-config",
                "--function-name",
                function_name,
                "--auth-type",
                "NONE",
                "--cors",
                json.dumps(
                    {
                        "AllowOrigins": ["*"],
                        "AllowMethods": ["GET"],
                        "AllowHeaders": ["*"],
                    }
                ),
                "--output",
                "json",
            ],
            region=region,
        )

    for statement_id, action in [
        ("function-url-public-access", "lambda:InvokeFunctionUrl"),
        ("function-public-access", "lambda:InvokeFunction"),
    ]:
        cmd = [
            "aws",
            "--region",
            region,
            "lambda",
            "add-permission",
            "--function-name",
            function_name,
            "--statement-id",
            statement_id,
            "--action",
            action,
            "--principal",
            "*",
        ]
        if action == "lambda:InvokeFunctionUrl":
            cmd.extend(["--function-url-auth-type", "NONE"])
        try:
            run(cmd, capture=False)
        except subprocess.CalledProcessError:
            pass

    state["function_name"] = function_name
    state["function_url"] = url_cfg["FunctionUrl"]
    save_state(state)
    return state


def main() -> None:
    load_env_file(ROOT / ".env.aws.local")
    load_env_file(ROOT / ".env.deploy.local")
    region = os.environ["AWS_REGION"]
    build_zip()
    role_arn = ensure_role("fashiongen-demo-lambda-role", region)
    state = ensure_function(region, role_arn)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
