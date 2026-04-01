Deploy scripts.

Frontend:

```bash
cd /Users/woodygoodenough/CodingProjects/FashionGenFineTuning
python3 deploy/deploy_frontend.py
```

This script:
- exports the demo catalog and thumbnails
- builds the static Next.js app
- syncs the site to S3
- reuses or provisions CloudFront and frontend DNS

Embedding backend:

```bash
cd /Users/woodygoodenough/CodingProjects/FashionGenFineTuning
python3 python/scripts/build_demo_embedding_index.py \
  --catalog nextjs/public/demo/catalog.json \
  --images-root nextjs/public/demo/thumbnails \
  --checkpoint /path/to/joint_checkpoint.pt \
  --output-dir deploy/.state/demo_index_10k
python3 deploy/deploy_embedding_api_ec2.py
```

Inputs can be overridden with:
- `DEMO_INDEX_DIR_LOCAL`
- `DEMO_MODEL_CHECKPOINT_LOCAL`

The backend deploy packages:
- FastAPI service
- precomputed image embeddings
- catalog metadata
- selected CLIP checkpoint

and publishes them behind the custom API hostname.
