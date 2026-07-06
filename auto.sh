#!/bin/bash

# 1. 关键：初始化 shell 里的 conda 环境（根据你服务器 anaconda 的安装路径修改）
# 通常在你的 ~/.bashrc 里能找到这段逻辑，但在脚本里必须显式写出
source /opt/anaconda3/etc/profile.d/conda.sh

# 2. 现在可以安全激活环境了
conda activate omnimotion

# 3. 设置数据路径
export DATA_DIR=/mnt/home/caixiang/4.12-carMove
export CUDA_VISIBLE_DEVICES=0

# 4. 运行预处理（确保在 preprocessing 目录下）
cd preprocessing
python main_processing.py --data_dir $DATA_DIR --chain

# 5. 运行训练
cd ..
python train.py --config configs/default.txt --data_dir $DATA_DIR