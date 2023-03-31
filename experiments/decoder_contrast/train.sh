#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
PARTITION=$1
JOB_NAME=$2
ROOT=../..
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$ROOT:$PYTHONPATH


python -m torch.distributed.launch --nproc_per_node=1 --master_port 29522 ../../train_contrast.py --config=config.yaml  2>&1 | tee log_$now.log
# mkdir -p checkpoints/result
# python ../../eval.py --base_size=512 --scales 1.0 --config=config.yaml --model_path=checkpoints/best.pth --save_folder=checkpoints/result/ 2>&1 | tee checkpoints/result/eva_$now.log
# python ../../eval.py --base_size=512 --scales 1.0 --config=config.yaml --model_path=checkpoints/epoch_99.pth --save_folder=checkpoints/result/ 2>&1 | tee checkpoints/result/eva_99_$now.log
