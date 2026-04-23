#!/bin/bash

# Stop the script immediately if any error occurs
set -e

echo "================================================================="
echo " CROSS-DOMAIN DIFFUSION RECSYS - END-TO-END TRAINING STARTS"
echo "================================================================="

# Set PYTHONPATH to the root directory so modules can find each other
export PYTHONPATH=$(pwd)

# --- SETTINGS ---
# You can change these variables based on the dataset you want to run
DATASET="douban" 
EMBED_DIM=256
GNN_EPOCHS=100

echo ""
echo "[STAGE 1] Training LightGCN Model..."
# Note: You can train with combined or target domain data.
python lightgcn/train.py \
    --dataset_path inters/DoubanMovie.train.inter \
    --embedding_dim $EMBED_DIM \
    --n_epochs $GNN_EPOCHS \
    --batch_size 2048 \
    --learning_rate 0.001

# Find the latest trained checkpoint
# This command dynamically fetches the most recently modified LightGCN checkpoint
GNN_CHECKPOINT=$(ls -t checkpoints/LightGCN*.pt | head -1)

echo ""
echo "[STAGE 2] Exporting LightGCN Embeddings for Diffusion (Bridge)..."
python lightgcn/export_for_conditioner.py \
    --checkpoint "$GNN_CHECKPOINT" \
    --dataset_path inters/DoubanMovie.train.inter \
    --output_path assets/lightgcn_dim${EMBED_DIM}.pt \
    --domain_items movie=inters/DoubanMovie.train.inter \
    --domain_items music=inters/DoubanMusic.train.inter 

echo ""
echo "[STAGE 3] Training End-to-End Diffusion Model..."
# Runs according to the settings in config.yaml
# We enter the diffusion directory to ensure relative module imports work properly
cd diffusion
python train.py
cd ..

echo ""
echo "[STAGE 4] Calculating Final Metrics on Test Data..."
cd diffusion
python test.py
cd ..

echo ""
echo "[STAGE 5] Generating Sample Recommendations for Target User..."
# Reads the inference -> target_user_id value from config.yaml
cd diffusion
python predict.py
cd ..

echo ""
echo "================================================================="
echo " ALL PROCESSES COMPLETED SUCCESSFULLY!"
echo "================================================================="