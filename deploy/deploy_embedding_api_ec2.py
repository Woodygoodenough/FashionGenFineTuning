#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import textwrap
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "deploy" / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "embedding_api_ec2.json"


def resolve_index_dir() -> Path:
    configured = os.environ.get("DEMO_INDEX_DIR_LOCAL")
    if configured:
        return Path(configured)
    return STATE_DIR / "demo_index_10k"


def resolve_checkpoint_path() -> Path:
    configured = os.environ.get("DEMO_MODEL_CHECKPOINT_LOCAL")
    if configured:
        return Path(configured)
    candidates = [
        ROOT / "report_update" / "remote_results" / "checkpoints" / "joint_w0p08_step3726.pt",
        ROOT / "artifacts" / "results" / "checkpoints" / "joint_w0p08_step3726.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve local joint checkpoint for embedding backend deploy")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def run(cmd: list[str], *, capture: bool = True) -> str:
    result = subprocess.run(cmd, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def aws(args: list[str], *, region: str) -> dict | list | str:
    out = run(["aws", "--region", region, *args])
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def ensure_bucket(bucket: str, region: str) -> None:
    try:
        aws(["s3api", "head-bucket", "--bucket", bucket], region=region)
    except subprocess.CalledProcessError:
        cmd = ["s3api", "create-bucket", "--bucket", bucket]
        if region != "us-east-1":
            cmd += ["--create-bucket-configuration", f"LocationConstraint={region}"]
        aws(cmd, region=region)


def create_bundle(bundle_path: Path) -> None:
    index_dir = resolve_index_dir()
    checkpoint_path = resolve_checkpoint_path()
    files = [
        (ROOT / "python" / "app" / "__init__.py", "python/app/__init__.py"),
        (ROOT / "python" / "app" / "demo_service.py", "python/app/demo_service.py"),
        (ROOT / "python" / "app" / "embedding_search.py", "python/app/embedding_search.py"),
        (ROOT / "python" / "app" / "demo_search.py", "python/app/demo_search.py"),
        (ROOT / "deploy" / "backend_api_requirements.txt", "deploy/backend_api_requirements.txt"),
        (index_dir / "catalog.json", "runtime/catalog.json"),
        (index_dir / "image_embeddings.npy", "runtime/image_embeddings.npy"),
        (checkpoint_path, f"runtime/{checkpoint_path.name}"),
    ]
    with tarfile.open(bundle_path, "w:gz") as tar:
        for src, arcname in files:
            tar.add(src, arcname=arcname)


def ensure_role(role_name: str, profile_name: str, region: str) -> str:
    try:
        role = aws(["iam", "get-role", "--role-name", role_name, "--output", "json"], region=region)["Role"]
        return role["Arn"]
    except subprocess.CalledProcessError:
        assume_role = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
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
        run(["aws", "iam", "attach-role-policy", "--role-name", role_name, "--policy-arn", "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"], capture=False)
        run(["aws", "iam", "attach-role-policy", "--role-name", role_name, "--policy-arn", "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"], capture=False)
        try:
            aws(["iam", "create-instance-profile", "--instance-profile-name", profile_name, "--output", "json"], region=region)
        except subprocess.CalledProcessError:
            pass
        time.sleep(5)
        try:
            run(["aws", "iam", "add-role-to-instance-profile", "--instance-profile-name", profile_name, "--role-name", role_name], capture=False)
        except subprocess.CalledProcessError:
            pass
        time.sleep(10)
        return role["Arn"]


def ensure_keypair(name: str, pem_path: Path, region: str) -> None:
    if pem_path.exists():
        return
    key = aws(["ec2", "create-key-pair", "--key-name", name, "--query", "KeyMaterial", "--output", "text"], region=region)
    pem_path.write_text(str(key))
    pem_path.chmod(0o600)


def ensure_security_groups(vpc_id: str, region: str) -> tuple[str, str]:
    def find(name: str) -> str | None:
        result = aws(
            ["ec2", "describe-security-groups", "--filters", f"Name=vpc-id,Values={vpc_id}", f"Name=group-name,Values={name}", "--output", "json"],
            region=region,
        )["SecurityGroups"]
        return result[0]["GroupId"] if result else None

    alb_name = "fashiongen-demo-alb-sg"
    ec2_name = "fashiongen-demo-ec2-sg"

    alb_sg = find(alb_name)
    if alb_sg is None:
        alb_sg = aws(["ec2", "create-security-group", "--group-name", alb_name, "--description", "ALB SG", "--vpc-id", vpc_id, "--output", "json"], region=region)["GroupId"]
        for port in ("80", "443"):
            try:
                run(["aws", "--region", region, "ec2", "authorize-security-group-ingress", "--group-id", alb_sg, "--protocol", "tcp", "--port", port, "--cidr", "0.0.0.0/0"], capture=False)
            except subprocess.CalledProcessError:
                pass

    ec2_sg = find(ec2_name)
    if ec2_sg is None:
        ec2_sg = aws(["ec2", "create-security-group", "--group-name", ec2_name, "--description", "EC2 SG", "--vpc-id", vpc_id, "--output", "json"], region=region)["GroupId"]
        try:
            run(["aws", "--region", region, "ec2", "authorize-security-group-ingress", "--group-id", ec2_sg, "--protocol", "tcp", "--port", "22", "--cidr", "0.0.0.0/0"], capture=False)
        except subprocess.CalledProcessError:
            pass
        try:
            run(["aws", "--region", region, "ec2", "authorize-security-group-ingress", "--group-id", ec2_sg, "--protocol", "tcp", "--port", "8000", "--source-group", alb_sg], capture=False)
        except subprocess.CalledProcessError:
            pass

    return alb_sg, ec2_sg


def ensure_certificate(domain: str, zone_id: str, token: str) -> str:
    existing = aws(["acm", "list-certificates", "--certificate-statuses", "PENDING_VALIDATION", "ISSUED", "--output", "json"], region="us-east-1")
    for cert in existing.get("CertificateSummaryList", []):
        if cert.get("DomainName") == domain:
            return cert["CertificateArn"]

    requested = aws(["acm", "request-certificate", "--domain-name", domain, "--validation-method", "DNS", "--output", "json"], region="us-east-1")
    cert_arn = requested["CertificateArn"]
    while True:
        details = aws(["acm", "describe-certificate", "--certificate-arn", cert_arn, "--output", "json"], region="us-east-1")["Certificate"]
        options = details.get("DomainValidationOptions", [])
        if options and options[0].get("ResourceRecord"):
            record = options[0]["ResourceRecord"]
            upsert_dns_record(zone_id, token, record["Type"], record["Name"], record["Value"], False)
            return cert_arn
        time.sleep(5)


def wait_for_certificate(cert_arn: str) -> None:
    while True:
        details = aws(["acm", "describe-certificate", "--certificate-arn", cert_arn, "--output", "json"], region="us-east-1")["Certificate"]
        status = details["Status"]
        if status == "ISSUED":
            return
        time.sleep(15)


def cf_request(method: str, path: str, token: str, payload: dict | None = None) -> dict:
    import urllib.request

    url = f"https://api.cloudflare.com/client/v4{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upsert_dns_record(zone_id: str, token: str, record_type: str, name: str, content: str, proxied: bool) -> None:
    current = cf_request("GET", f"/zones/{zone_id}/dns_records?type={record_type}&name={name}", token)
    payload = {"type": record_type, "name": name, "content": content, "proxied": proxied, "ttl": 1}
    if current["result"]:
        record_id = current["result"][0]["id"]
        cf_request("PUT", f"/zones/{zone_id}/dns_records/{record_id}", token, payload)
    else:
        cf_request("POST", f"/zones/{zone_id}/dns_records", token, payload)


def latest_amazon_linux_ami(region: str) -> str:
    return str(
        aws(
            [
                "ssm",
                "get-parameter",
                "--name",
                "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64",
                "--query",
                "Parameter.Value",
                "--output",
                "text",
            ],
            region=region,
        )
    )


def render_user_data(bucket: str, object_key: str, checkpoint_name: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        set -euxo pipefail
        dnf update -y
        dnf install -y python3.11 python3.11-pip python3.11-devel git gcc gcc-c++ make
        mkdir -p /opt/fashiongen-demo
        aws s3 cp s3://{bucket}/{object_key} /opt/fashiongen-demo/backend_bundle.tar.gz
        tar -xzf /opt/fashiongen-demo/backend_bundle.tar.gz -C /opt/fashiongen-demo
        python3.11 -m venv /opt/fashiongen-demo/.venv
        source /opt/fashiongen-demo/.venv/bin/activate
        python -m pip install --upgrade pip
        python -m pip install -r /opt/fashiongen-demo/deploy/backend_api_requirements.txt
        cat >/etc/systemd/system/fashiongen-demo-api.service <<'EOF'
        [Unit]
        Description=FashionGen Demo API
        After=network.target

        [Service]
        Type=simple
        WorkingDirectory=/opt/fashiongen-demo
        Environment=DEMO_SEARCH_MODE=embedding
        Environment=DEMO_INDEX_DIR=/opt/fashiongen-demo/runtime
        Environment=DEMO_MODEL_CHECKPOINT=/opt/fashiongen-demo/runtime/{checkpoint_name}
        Environment=DEMO_MODEL_NAME=ViT-B-16
        Environment=DEMO_MODEL_PRETRAINED=laion2b_s34b_b88k
        ExecStart=/opt/fashiongen-demo/.venv/bin/uvicorn python.app.demo_service:app --host 0.0.0.0 --port 8000
        Restart=always
        RestartSec=3

        [Install]
        WantedBy=multi-user.target
        EOF
        systemctl daemon-reload
        systemctl enable fashiongen-demo-api
        systemctl restart fashiongen-demo-api
        """
    )


def main() -> None:
    load_env_file(ROOT / ".env.aws.local")
    load_env_file(ROOT / ".env.deploy.local")
    region = os.environ["AWS_REGION"]
    zone_id = os.environ["CLOUDFLARE_ZONE_ID"]
    cf_token = os.environ["CLOUDFLARE_API_TOKEN"]
    api_host = os.environ.get("API_HOST", "api.woodygoodenough.com")

    state = load_state()
    identity = aws(["sts", "get-caller-identity", "--output", "json"], region=region)
    account_id = identity["Account"]
    backend_bucket = f"fashiongen-demo-backend-{account_id}"
    ensure_bucket(backend_bucket, region)

    bundle_path = STATE_DIR / "backend_bundle.tar.gz"
    create_bundle(bundle_path)
    object_key = "backend/backend_bundle.tar.gz"
    run(["aws", "--region", region, "s3", "cp", str(bundle_path), f"s3://{backend_bucket}/{object_key}"], capture=False)

    vpc_id = str(aws(["ec2", "describe-vpcs", "--query", "Vpcs[0].VpcId", "--output", "text"], region=region))
    subnets = aws(["ec2", "describe-subnets", "--filters", f"Name=vpc-id,Values={vpc_id}", "--query", "Subnets[].SubnetId", "--output", "json"], region=region)
    subnet_ids = [str(s) for s in subnets][:2]
    alb_sg, ec2_sg = ensure_security_groups(vpc_id, region)

    key_name = "fashiongen-demo-ec2-key"
    key_path = STATE_DIR / f"{key_name}.pem"
    ensure_keypair(key_name, key_path, region)

    role_name = "fashiongen-demo-ec2-role"
    profile_name = "fashiongen-demo-ec2-profile"
    ensure_role(role_name, profile_name, region)

    instance_id = state.get("instance_id")
    if not instance_id:
        ami_id = latest_amazon_linux_ami(region)
        user_data = render_user_data(backend_bucket, object_key, resolve_checkpoint_path().name)
        launched = aws(
            [
                "ec2",
                "run-instances",
                "--image-id",
                ami_id,
                "--instance-type",
                os.environ.get("DEMO_BACKEND_INSTANCE_TYPE", "m7i-flex.large"),
                "--iam-instance-profile",
                f"Name={profile_name}",
                "--key-name",
                key_name,
                "--security-group-ids",
                ec2_sg,
                "--subnet-id",
                subnet_ids[0],
                "--user-data",
                user_data,
                "--tag-specifications",
                "ResourceType=instance,Tags=[{Key=Name,Value=fashiongen-demo-api}]",
                "--query",
                "Instances[0].InstanceId",
                "--output",
                "text",
            ],
            region=region,
        )
        instance_id = str(launched)
        state["instance_id"] = instance_id
        save_state(state)

    run(["aws", "--region", region, "ec2", "wait", "instance-running", "--instance-ids", instance_id], capture=False)
    desc = aws(["ec2", "describe-instances", "--instance-ids", instance_id, "--output", "json"], region=region)
    instance = desc["Reservations"][0]["Instances"][0]
    instance_ip = instance.get("PublicIpAddress")
    state["instance_public_ip"] = instance_ip

    cert_arn = ensure_certificate(api_host, zone_id, cf_token)
    wait_for_certificate(cert_arn)

    target_group_arn = state.get("target_group_arn")
    if not target_group_arn:
        target_group_arn = aws(
            [
                "elbv2",
                "create-target-group",
                "--name",
                "fashiongen-demo-tg",
                "--protocol",
                "HTTP",
                "--port",
                "8000",
                "--vpc-id",
                vpc_id,
                "--target-type",
                "instance",
                "--health-check-path",
                "/api/health",
                "--query",
                "TargetGroups[0].TargetGroupArn",
                "--output",
                "text",
            ],
            region=region,
        )
        state["target_group_arn"] = str(target_group_arn)
        save_state(state)

    run(["aws", "--region", region, "elbv2", "register-targets", "--target-group-arn", str(target_group_arn), "--targets", f"Id={instance_id},Port=8000"], capture=False)

    alb_arn = state.get("alb_arn")
    alb_dns = state.get("alb_dns")
    if not alb_arn:
        created = aws(
            [
                "elbv2",
                "create-load-balancer",
                "--name",
                "fashiongen-demo-alb",
                "--subnets",
                *subnet_ids,
                "--security-groups",
                alb_sg,
                "--scheme",
                "internet-facing",
                "--type",
                "application",
                "--query",
                "LoadBalancers[0]",
                "--output",
                "json",
            ],
            region=region,
        )
        alb_arn = created["LoadBalancerArn"]
        alb_dns = created["DNSName"]
        state["alb_arn"] = alb_arn
        state["alb_dns"] = alb_dns
        save_state(state)

    listeners = aws(["elbv2", "describe-listeners", "--load-balancer-arn", alb_arn, "--output", "json"], region=region)["Listeners"]
    ports = {listener["Port"] for listener in listeners}
    if 80 not in ports:
        run(["aws", "--region", region, "elbv2", "create-listener", "--load-balancer-arn", alb_arn, "--protocol", "HTTP", "--port", "80", "--default-actions", f"Type=redirect,RedirectConfig={{Protocol=HTTPS,Port=443,StatusCode=HTTP_301}}"], capture=False)
    if 443 not in ports:
        run(["aws", "--region", region, "elbv2", "create-listener", "--load-balancer-arn", alb_arn, "--protocol", "HTTPS", "--port", "443", "--certificates", f"CertificateArn={cert_arn}", "--ssl-policy", "ELBSecurityPolicy-TLS13-1-2-2021-06", "--default-actions", f"Type=forward,TargetGroupArn={target_group_arn}"], capture=False)

    upsert_dns_record(zone_id, cf_token, "CNAME", api_host, str(alb_dns), True)
    state["api_host"] = api_host
    state["backend_bucket"] = backend_bucket
    save_state(state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
