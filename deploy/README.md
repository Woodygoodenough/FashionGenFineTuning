Deploy flow:

1. Set secrets in `.env.aws.local` and `.env.deploy.local`.
2. Run:

```bash
cd /Users/woodygoodenough/CodingProjects/FashionGenFineTuning
python3 deploy/deploy_frontend.py
```

Frontend deploy script:
- builds the static Next.js frontend
- exports a 10,000-image demo catalog
- provisions or reuses ACM, S3, and CloudFront
- publishes the site
- upserts the Cloudflare DNS record for the frontend host

Backend deploy:

```bash
cd /Users/woodygoodenough/CodingProjects/FashionGenFineTuning
python3 deploy/deploy_demo_api.py
```

Backend deploy script:
- packages the lightweight demo search API for Lambda
- creates or updates the Lambda function URL
- reuses the generated 10,000-item catalog
