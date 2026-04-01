Frontend deploy flow:

1. Set secrets in `.env.aws.local` and `.env.deploy.local`.
2. Run:

```bash
cd /Users/woodygoodenough/CodingProjects/FashionGenFineTuning
set -a
source .env.aws.local
source .env.deploy.local
set +a
python3 deploy/deploy_frontend.py
```

This script:
- builds the static Next.js frontend
- provisions or reuses ACM, S3, and CloudFront
- publishes the site
- upserts the Cloudflare DNS record for the frontend host
