#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LaTeX assets from artifacts/results."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def macro(name: str, value: str) -> str:
    return f"\\renewcommand{{\\{name}}}{{{value}}}"


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    art = repo / "artifacts" / "results"
    build = repo / "reports" / "build"
    build.mkdir(parents=True, exist_ok=True)

    zero = json.loads((art / "retrieval" / "zero_shot_vs_finetuned.json").read_text())
    ctrl = json.loads((art / "retrieval" / "controlled_comparison.json").read_text())
    ann = json.loads((art / "ann" / "ann_compare_30k_selected.json").read_text())
    tsne = json.loads((art / "tsne" / "fashiongen_tsne_triplet_summary.json").read_text())

    zero_shot = zero["zero_shot"]
    finetuned = zero["finetuned"]
    retrieval = ctrl["retrieval_only"]["best_eval"]
    joint = ctrl["joint_shared"]["best_eval"]

    lines = [
        macro("ZeroAvgRone", fmt(zero_shot["avg_r1"])),
        macro("ZeroAvgRfive", fmt(zero_shot["avg_r5"])),
        macro("ZeroAvgRten", fmt(zero_shot["avg_r10"])),
        macro("FTAvgRone", fmt(finetuned["avg_r1"])),
        macro("FTAvgRfive", fmt(finetuned["avg_r5"])),
        macro("FTAvgRten", fmt(finetuned["avg_r10"])),
        macro("DeltaAvgRten", f"{finetuned['avg_r10'] - zero_shot['avg_r10']:+.4f}"),
        macro("RetAvgRone", fmt(retrieval["avg_r1"])),
        macro("RetAvgRfive", fmt(retrieval["avg_r5"])),
        macro("RetAvgRten", fmt(retrieval["avg_r10"])),
        macro("JointAvgRone", fmt(joint["avg_r1"])),
        macro("JointAvgRfive", fmt(joint["avg_r5"])),
        macro("JointAvgRten", fmt(joint["avg_r10"])),
        macro("JointBaselineScore", f"{retrieval['score']:.4f}"),
        macro("JointBestScore", f"{joint['score']:.4f}"),
        macro("JointBestWeight", str(ctrl["config"]["joint_cls_weight"])),
        macro("JointDeltaScore", f"{joint['score'] - retrieval['score']:+.4f}"),
        macro("AnnExactLarge", f"{ann['exact']['avg_ms']:.3f}"),
        macro("AnnApproxLarge", f"{ann['pure_ann']['avg_ms']:.3f}"),
        macro("AnnOverlapLarge", f"{ann['pure_ann']['overlap_at_k']:.3f}"),
        macro("AnnSpeedupLarge", f"{ann['pure_ann']['speedup']:.2f}$\\times$"),
        macro("ZeroKNNFive", fmt(tsne["zero_shot"]["knn_same_category_at5"])),
        macro("KNNBase", fmt(tsne["zero_shot"]["knn_same_category_at10"])),
        macro("ZeroSilhouette", fmt(tsne["zero_shot"]["silhouette"])),
        macro("ZeroDBI", fmt(tsne["zero_shot"]["davies_bouldin"])),
        macro("RetKNNFive", fmt(tsne["retrieval_only"]["knn_same_category_at5"])),
        macro("RetKNNTen", fmt(tsne["retrieval_only"]["knn_same_category_at10"])),
        macro("RetSilhouette", fmt(tsne["retrieval_only"]["silhouette"])),
        macro("RetDBI", fmt(tsne["retrieval_only"]["davies_bouldin"])),
        macro("JointKNNFive", fmt(tsne["joint"]["knn_same_category_at5"])),
        macro("KNNFine", fmt(tsne["joint"]["knn_same_category_at10"])),
        macro("JointSilhouette", fmt(tsne["joint"]["silhouette"])),
        macro("JointDBI", fmt(tsne["joint"]["davies_bouldin"])),
    ]

    (build / "results_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tsne_plot = art / "tsne" / "fashiongen_tsne_triplet.png"
    if tsne_plot.exists():
        shutil.copy2(tsne_plot, build / "fashiongen_tsne_triplet.png")
    summary = {
        "retrieval": zero,
        "controlled_comparison": ctrl,
        "ann": ann,
        "tsne": tsne,
    }
    (build / "report_asset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
