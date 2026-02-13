#!/bin/bash
# Launch parallel training on both RTX 5090 GPUs
# Usage: ./launch_training.sh [fold]

FOLD=${1:-0}
PATCH_SIZE=${2:-64}

DATA_DIR="data/patches_roi"
LABELS_CSV="data/train_labels_14class.csv"
CV_DIR="data/cv_splits"
LOG_DIR="logs"
MODEL_DIR="models"

mkdir -p $LOG_DIR
mkdir -p $MODEL_DIR

echo "="
echo "LAUNCHING PARALLEL TRAINING"
echo "="
echo "Fold: $FOLD"
echo "Patch size: ${PATCH_SIZE}^3"
echo "="

# GPU 0 - ResNet-18
echo "Starting ResNet-18 on GPU 0..."
env CUDA_VISIBLE_DEVICES=0 nohup bash -c "conda activate rsna_kaggle && python scripts/04_train_model.py \
 --data-dir $DATA_DIR \
 --labels-csv $LABELS_CSV \
 --cv-dir $CV_DIR \
 --fold $FOLD \
 --arch resnet18 \
 --patch-size $PATCH_SIZE \
 --output ${MODEL_DIR}/fold${FOLD}_resnet18_p${PATCH_SIZE} \
 --epochs 50 \
 --batch-size 10 \
 --lr 0.001 \
 --augment" \
 > ${LOG_DIR}/train_fold${FOLD}_resnet18_p${PATCH_SIZE}.log 2>&1 &

GPU0_PID=$!
echo " PID: $GPU0_PID"

# Wait a few seconds to avoid I/O conflicts
sleep 5

# GPU 1 - DenseNet-121
echo "Starting DenseNet-121 on GPU 1..."
env CUDA_VISIBLE_DEVICES=1 nohup bash -c "conda activate rsna_kaggle && python scripts/04_train_model.py \
 --data-dir $DATA_DIR \
 --labels-csv $LABELS_CSV \
 --cv-dir $CV_DIR \
 --fold $FOLD \
 --arch densenet121 \
 --patch-size $PATCH_SIZE \
 --output ${MODEL_DIR}/fold${FOLD}_densenet121_p${PATCH_SIZE} \
 --epochs 50 \
 --batch-size 6 \
 --lr 0.001 \
 --augment" \
 > ${LOG_DIR}/train_fold${FOLD}_densenet121_p${PATCH_SIZE}.log 2>&1 &

GPU1_PID=$!
echo " PID: $GPU1_PID"

echo ""
echo "Training launched on both GPUs!"
echo "Monitor with:"
echo " GPU 0: tail -f ${LOG_DIR}/train_fold${FOLD}_resnet18_p${PATCH_SIZE}.log"
echo " GPU 1: tail -f ${LOG_DIR}/train_fold${FOLD}_densenet121_p${PATCH_SIZE}.log"
echo ""
echo "Check GPU usage: nvidia-smi"
echo "Check processes: ps aux | grep 04_train_model"
echo ""
echo "Kill all training:"
echo " kill $GPU0_PID $GPU1_PID"
