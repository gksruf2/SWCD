#!/bin/bash

#SBATCH --job-name qwer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -o logss/slurm-%A-%x.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/opt/anaconda3/envs/pytorch1.12.1_p38/lib/python3.8/site-packages/torch/lib

#python data/bair/convert_bair.py --data_dir /data/gksruf293/swcd/image2video-synthesis-using-cINNs-main/download_data --output_dir /data/gksruf293/swcd/image2video-synthesis-using-cINNs-main/download_bair
#python -W ignore generate_samples.py -dataset bair -gpu 0

wandb login 0d96660e1ccd97c40207a831c91de88b8763fa7b
python -W ignore -m stage1_VAE.main_copy -gpu 0
#python -W ignore -m stage2_cINN.AE.main -gpu 0 -cf stage2_cINN/AE/configs/bair_config.yaml
#python -W ignore -m stage2_cINN.main -gpu 0 -cf stage2_cINN/configs/bair_config.yaml

exit 0