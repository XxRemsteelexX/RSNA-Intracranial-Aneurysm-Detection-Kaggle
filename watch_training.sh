#!/bin/bash
# Monitor training progress on both GPUs
# Usage: ./watch_training.sh [fold] [patch_size]

FOLD=${1:-0}
PATCH_SIZE=${2:-64}

LOG_DIR="logs"
GPU0_LOG="${LOG_DIR}/train_fold${FOLD}_resnet34_p${PATCH_SIZE}.log"
GPU1_LOG="${LOG_DIR}/train_fold${FOLD}_resnet50_p${PATCH_SIZE}.log"

clear
echo "========================================"
echo "TRAINING MONITOR - Fold $FOLD, Patch ${PATCH_SIZE}^3"
echo "========================================"
echo ""

# GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r idx name util mem_used mem_total; do
 echo " GPU $idx: $name | Util: ${util}% | VRAM: ${mem_used}MB / ${mem_total}MB"
done
echo ""

# GPU 0 (ResNet-34)
echo "----------------------------------------"
echo "GPU 0 - ResNet-34"
echo "----------------------------------------"
if [ -f "$GPU0_LOG" ]; then
 # Extract latest epoch info
 tail -50 "$GPU0_LOG" | grep -E "(Epoch|Train Loss|Val Loss|Val AUC|best AUC)" | tail -10
 echo ""
 echo "Latest lines:"
 tail -3 "$GPU0_LOG"
else
 echo "Log not found: $GPU0_LOG"
fi

echo ""
echo "----------------------------------------"
echo "GPU 1 - ResNet-50"
echo "----------------------------------------"
if [ -f "$GPU1_LOG" ]; then
 # Extract latest epoch info
 tail -50 "$GPU1_LOG" | grep -E "(Epoch|Train Loss|Val Loss|Val AUC|best AUC)" | tail -10
 echo ""
 echo "Latest lines:"
 tail -3 "$GPU1_LOG"
else
 echo "Log not found: $GPU1_LOG"
fi

echo ""
echo "========================================"
echo "Press Ctrl+C to exit"
echo "Refresh every 10 seconds with: watch -n 10 ./watch_training.sh $FOLD $PATCH_SIZE"
