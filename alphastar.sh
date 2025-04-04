#!/bin/bash

echo "Setting up AlphaStar Tim (2025) Environment..."
export ALPHA_STAR_DIR="/mnt/Data2/workspace/AlphaStar/alphastar_tim"
source activate base	
conda deactivate
conda activate conda39-alphastar-tim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export SC2PATH="/mnt/Data2/workspace/AlphaStar/StarCraftII"
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib"
echo "$ALPHA_STAR_DIR"
