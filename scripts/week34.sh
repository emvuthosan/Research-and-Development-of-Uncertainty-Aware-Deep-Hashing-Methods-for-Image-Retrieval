#!/usr/bin/env bash
# Week 3-4 runs (ProbHash + ablation, Table 2 of the paper). Run from the repo root:
#   bash scripts/week34.sh A            # one job (A, B, C, D, smoke, eval100, report)
#   bash scripts/week34.sh A B          # several jobs one after another
#   DATASET=nuswide_21 BIT="16 32 64" bash scripts/week34.sh A
# Colab: one job per notebook/GPU, then copy saved_model/<run>/ back to Drive.
set -euo pipefail

DATASET=${DATASET:-imagenet}
BIT=${BIT:-16}
EPOCH=${EPOCH:-100}
SEED=${SEED:-0}
WORKERS=${WORKERS:-4}
COMMON="--dataset $DATASET --bit $BIT --epoch $EPOCH --seed $SEED --num_workers $WORKERS --val --no-pairwise --no-tqdm --rt_st ham --sample 1"

run() { echo ">>> $*"; "$@"; }

for job in "${@:-help}"; do
  case "$job" in
    smoke)   # 10 epochs, just checks that the pipeline runs end to end
             run python CSQ_HashUQ.py --grad_est cf --KL_regularization 1 --dataset $DATASET --bit 16 --epoch 10 \
                 --seed 99 --num_workers $WORKERS --val --no-pairwise --no-tqdm --rt_st ham --sample 1 ;;
    A)       run python CSQ_HashUQ.py        --grad_est cf --KL_regularization 1 $COMMON ;;   # ProbHash Center (closed-form)
    B)       run python CSQ_HashUQ.py        --grad_est st --KL_regularization 1 $COMMON ;;   # ablation: straight-through, lambda=1
    C)       run python CSQ_HashUQ.py        --grad_est st --KL_regularization 0 $COMMON ;;   # ablation: straight-through, lambda=0
    D)       run python GreedyHash_HashUQ.py --grad_est st --KL_regularization 1 $COMMON ;;   # ProbHash Label (ST)
    eval100) # re-evaluate the best checkpoints of A and D with 100 MC Dropout samples (no retraining)
             E="--dataset $DATASET --bit $BIT --epoch $EPOCH --seed $SEED --num_workers $WORKERS --val --no-pairwise --no-tqdm --rt_st hamuct --sample 100 --sample_method MCD --no-train"
             run python CSQ_HashUQ.py        --grad_est cf --KL_regularization 1 $E
             run python GreedyHash_HashUQ.py --grad_est st --KL_regularization 1 $E ;;
    report)  mkdir -p results
             run python summarize_results.py saved_model --out results/summary
             run python plot_curves.py saved_model --filter "dataset_$DATASET" --out results/curves_$DATASET.png ;;
    *)       sed -n '2,6p' "$0"; exit 1 ;;
  esac
done
