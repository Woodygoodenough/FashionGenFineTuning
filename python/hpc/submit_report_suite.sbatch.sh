#!/bin/bash
#SBATCH --job-name=fashiongen-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${CONFIG_PATH:?set CONFIG_PATH}"
: "${PYTHON_BIN:=python}"
: "${TASKS:=compare evaluate tsne ann}"

cd "$PROJECT_ROOT/python"
$PYTHON_BIN scripts/run_report_suite.py --config "$CONFIG_PATH" --tasks $TASKS
