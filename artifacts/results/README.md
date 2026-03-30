# Experiment Results

This directory contains the retained raw experiment outputs used for reproducibility.

Retention rule:

- keep JSON and text outputs
- regenerate plots and other derived visual assets from code when needed

- `retrieval/zero_shot_vs_finetuned.*`: zero-shot vs FashionGen fine-tuned evaluation on the same validation subset.
- `retrieval/controlled_comparison.*`: controlled retrieval-only vs joint-loss comparison with matched hyperparameters.
- `ann/ann_compare_30k_selected.json`: ANN benchmark used in the retained benchmark set (`30k`, `efSearch=128`, `rerank_k=200`).
- `ann/tuning/*.json`: additional ANN tuning runs considered during selection.
- `tsne/fashiongen_tsne_triplet_summary.json`: three-way t-SNE clustering metrics.
- `tsne/fashiongen_tsne_manifest.json`: fixed balanced sample manifest used to regenerate the t-SNE plot.
