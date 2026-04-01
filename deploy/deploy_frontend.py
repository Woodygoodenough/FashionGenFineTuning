#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NEXTJS_DIR = REPO_ROOT / "nextjs"
STATE_DIR = REPO_ROOT / "deploy" / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "frontend.json"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = True) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def aws(cmd: list[str], *, region: str | None = None) -> dict | list | str:
    full = ["aws"]
    if region:
      full += ["--region", region]
    full += cmd
    out = run(full)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def cf_request(method: str, path: str, token: str, payload: dict | None = None) -> dict:
    url = f"https://api.cloudflare.com/client/v4{path}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloudflare API error {exc.code}: {body}") from exc


def load_required_env() -> dict[str, str]:
    keys = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ZONE_ID",
        "CLOUDFLARE_ZONE_NAME",
        "FRONTEND_HOST",
    ]
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing required env: {', '.join(missing)}")
    return {k: os.environ[k] for k in keys}


def ensure_certificate(domain: str, zone_id: str, token: str) -> str:
    existing = aws(
        [
            "acm",
            "list-certificates",
            "--certificate-statuses",
            "PENDING_VALIDATION",
            "ISSUED",
            "--output",
            "json",
        ],
        region="us-east-1",
    )
    for cert in existing.get("CertificateSummaryList", []):
        if cert.get("DomainName") == domain:
            return cert["CertificateArn"]

    requested = aws(
        [
            "acm",
            "request-certificate",
            "--domain-name",
            domain,
            "--validation-method",
            "DNS",
            "--output",
            "json",
        ],
        region="us-east-1",
    )
    cert_arn = requested["CertificateArn"]

    while True:
        details = aws(
            [
                "acm",
                "describe-certificate",
                "--certificate-arn",
                cert_arn,
                "--output",
                "json",
            ],
            region="us-east-1",
        )["Certificate"]
        options = details.get("DomainValidationOptions", [])
        if options and options[0].get("ResourceRecord"):
            record = options[0]["ResourceRecord"]
            upsert_dns_record(
                zone_id=zone_id,
                token=token,
                record_type=record["Type"],
                name=record["Name"],
                content=record["Value"],
                proxied=False,
            )
            return cert_arn
        time.sleep(5)


def wait_for_certificate(cert_arn: str) -> None:
    while True:
        details = aws(
            [
                "acm",
                "describe-certificate",
                "--certificate-arn",
                cert_arn,
                "--output",
                "json",
            ],
            region="us-east-1",
        )["Certificate"]
        status = details["Status"]
        if status == "ISSUED":
            return
        if status not in {"PENDING_VALIDATION", "INACTIVE"}:
            raise RuntimeError(f"Certificate failed with status {status}")
        time.sleep(15)


def ensure_bucket(bucket: str, region: str) -> None:
    try:
        aws(["s3api", "head-bucket", "--bucket", bucket], region=region)
    except subprocess.CalledProcessError:
        cmd = ["s3api", "create-bucket", "--bucket", bucket]
        if region != "us-east-1":
            cmd += ["--create-bucket-configuration", f"LocationConstraint={region}"]
        aws(cmd, region=region)

    aws(
        [
            "s3api",
            "put-public-access-block",
            "--bucket",
            bucket,
            "--public-access-block-configuration",
            "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false",
        ],
        region=region,
    )
    aws(
        [
            "s3api",
            "put-bucket-website",
            "--bucket",
            bucket,
            "--website-configuration",
            json.dumps(
                {
                    "IndexDocument": {"Suffix": "index.html"},
                    "ErrorDocument": {"Key": "index.html"},
                }
            ),
        ],
        region=region,
    )
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket}/*"],
            }
        ],
    }
    aws(
        [
            "s3api",
            "put-bucket-policy",
            "--bucket",
            bucket,
            "--policy",
            json.dumps(policy),
        ],
        region=region,
    )


def build_site() -> Path:
    run(
        [
            "python3",
            str(REPO_ROOT / "python" / "scripts" / "export_demo_catalog.py"),
            "--limit",
            os.environ.get("DEMO_IMAGE_LIMIT", "10000"),
        ],
        cwd=REPO_ROOT,
        capture=False,
    )
    run(["npm", "install"], cwd=NEXTJS_DIR, capture=False)
    run(["npm", "run", "build"], cwd=NEXTJS_DIR, capture=False)
    out_dir = NEXTJS_DIR / "out"
    if not out_dir.exists():
        raise RuntimeError("Static export missing: nextjs/out")
    return out_dir


def sync_site(out_dir: Path, bucket: str, region: str) -> None:
    run(
        [
            "aws",
            "--region",
            region,
            "s3",
            "sync",
            str(out_dir) + "/",
            f"s3://{bucket}/",
            "--delete",
            "--exclude",
            "demo/thumbnails/*",
        ],
        capture=False,
    )
    demo_dir = out_dir / "demo"
    if demo_dir.exists():
        for name in ["catalog.json", "manifest.txt"]:
            path = demo_dir / name
            if path.exists():
                run(
                    [
                        "aws",
                        "--region",
                        region,
                        "s3",
                        "cp",
                        str(path),
                        f"s3://{bucket}/demo/{name}",
                    ],
                    capture=False,
                )
        thumbs_dir = demo_dir / "thumbnails"
        if thumbs_dir.exists():
            run(
                [
                    "aws",
                    "--region",
                    region,
                    "s3",
                    "sync",
                    str(thumbs_dir) + "/",
                    f"s3://{bucket}/demo/thumbnails/",
                    "--delete",
                    "--size-only",
                ],
                capture=False,
            )


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def ensure_distribution(bucket: str, domain: str, cert_arn: str, state: dict) -> tuple[str, str]:
    existing_id = state.get("distribution_id")
    if existing_id:
        summary = aws(["cloudfront", "get-distribution", "--id", existing_id, "--output", "json"])
        distribution = summary["Distribution"]
        return existing_id, distribution["DomainName"]

    origin_domain = f"{bucket}.s3-website-us-east-1.amazonaws.com"
    config = {
        "CallerReference": f"fashiongen-{int(time.time())}",
        "Aliases": {"Quantity": 1, "Items": [domain]},
        "DefaultRootObject": "index.html",
        "Origins": {
            "Quantity": 1,
            "Items": [
                {
                    "Id": "s3-website-origin",
                    "DomainName": origin_domain,
                    "CustomOriginConfig": {
                        "HTTPPort": 80,
                        "HTTPSPort": 443,
                        "OriginProtocolPolicy": "http-only",
                        "OriginSslProtocols": {
                            "Quantity": 3,
                            "Items": ["TLSv1", "TLSv1.1", "TLSv1.2"],
                        },
                        "OriginReadTimeout": 30,
                        "OriginKeepaliveTimeout": 5,
                    },
                }
            ],
        },
        "DefaultCacheBehavior": {
            "TargetOriginId": "s3-website-origin",
            "ViewerProtocolPolicy": "redirect-to-https",
            "AllowedMethods": {"Quantity": 2, "Items": ["HEAD", "GET"], "CachedMethods": {"Quantity": 2, "Items": ["HEAD", "GET"]}},
            "Compress": True,
            "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
        },
        "CustomErrorResponses": {
            "Quantity": 2,
            "Items": [
                {"ErrorCode": 403, "ResponsePagePath": "/index.html", "ResponseCode": "200", "ErrorCachingMinTTL": 0},
                {"ErrorCode": 404, "ResponsePagePath": "/index.html", "ResponseCode": "200", "ErrorCachingMinTTL": 0},
            ],
        },
        "Comment": "FashionGen frontend demo",
        "Enabled": True,
        "PriceClass": "PriceClass_100",
        "ViewerCertificate": {
            "ACMCertificateArn": cert_arn,
            "SSLSupportMethod": "sni-only",
            "MinimumProtocolVersion": "TLSv1.2_2021",
            "Certificate": cert_arn,
            "CertificateSource": "acm",
        },
    }
    created = aws(
        [
            "cloudfront",
            "create-distribution",
            "--distribution-config",
            json.dumps(config),
            "--output",
            "json",
        ]
    )["Distribution"]
    state["distribution_id"] = created["Id"]
    save_state(state)
    return created["Id"], created["DomainName"]


def wait_for_distribution(distribution_id: str) -> None:
    while True:
        distribution = aws(
            ["cloudfront", "get-distribution", "--id", distribution_id, "--output", "json"]
        )["Distribution"]
        if distribution["Status"] == "Deployed":
            return
        time.sleep(20)


def invalidate_distribution(distribution_id: str) -> None:
    aws(
        [
            "cloudfront",
            "create-invalidation",
            "--distribution-id",
            distribution_id,
            "--paths",
            "/*",
            "--output",
            "json",
        ]
    )


def upsert_dns_record(
    *,
    zone_id: str,
    token: str,
    record_type: str,
    name: str,
    content: str,
    proxied: bool,
) -> None:
    current = cf_request("GET", f"/zones/{zone_id}/dns_records?type={record_type}&name={name}", token)
    payload = {"type": record_type, "name": name, "content": content, "proxied": proxied, "ttl": 1}
    if current["result"]:
        record_id = current["result"][0]["id"]
        cf_request("PUT", f"/zones/{zone_id}/dns_records/{record_id}", token, payload)
    else:
        cf_request("POST", f"/zones/{zone_id}/dns_records", token, payload)


def main() -> None:
    load_env_file(REPO_ROOT / ".env.aws.local")
    load_env_file(REPO_ROOT / ".env.deploy.local")
    env = load_required_env()
    identity = aws(["sts", "get-caller-identity", "--output", "json"], region=env["AWS_REGION"])
    account_id = identity["Account"]

    bucket = f"fashiongen-demo-site-{account_id}"
    domain = env["FRONTEND_HOST"]
    region = env["AWS_REGION"]

    print(f"Using AWS account {account_id}")
    print(f"Target host: {domain}")
    print(f"S3 bucket: {bucket}")

    out_dir = build_site()
    cert_arn = ensure_certificate(domain, env["CLOUDFLARE_ZONE_ID"], env["CLOUDFLARE_API_TOKEN"])
    print(f"Certificate requested/found: {cert_arn}")
    wait_for_certificate(cert_arn)
    print("Certificate issued")

    ensure_bucket(bucket, region)
    sync_site(out_dir, bucket, region)

    state = load_state()
    distribution_id, distribution_domain = ensure_distribution(bucket, domain, cert_arn, state)
    print(f"CloudFront distribution: {distribution_id} ({distribution_domain})")
    wait_for_distribution(distribution_id)
    invalidate_distribution(distribution_id)

    upsert_dns_record(
        zone_id=env["CLOUDFLARE_ZONE_ID"],
        token=env["CLOUDFLARE_API_TOKEN"],
        record_type="CNAME",
        name=domain,
        content=distribution_domain,
        proxied=True,
    )

    print(json.dumps({"frontend_host": domain, "distribution_id": distribution_id, "distribution_domain": distribution_domain, "bucket": bucket}, indent=2))


if __name__ == "__main__":
    main()
