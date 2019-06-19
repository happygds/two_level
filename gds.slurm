#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=k40
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=YOUR@MAIL
#SBATCH --output=out.out
#SBATCH --error=err.err
#SBATCH --time=32:15:00

source /usr/share/Modules/init/bash
module purge
module load gcc/4.9 miniconda3 cuda/8.0
source activate py36

# rm -r ../THUMOS14/frames
# python thumos_vid2jpg.py

cd broi1d_pooling_avg
python build.py
cd ..

cd broi1d_align
python build.py
cd ..

nvidia-smi

CUDA_VISIBLE_DEVICES=0 python binary_train.py thumos14 -b 32 --epochs 10 --n_layers 1 --pos_enc 0 --result_path ../THUMOS14/result_two_level/seed1 --num_local 0 --local_type qkv --dilated_mask 0 --seed 1 --groupwise_heads 5 --roi_poolsize 3 --feat_model feature_anet_200 && \
python binary_test.py activitynet1.3 validation ../THUMOS14/result_two_level/seed1/self_attn_N1_nopos_G5_roi3/model_best.pth.tar  ../THUMOS14/result_two_level/seed1/self_attn_N1_nopos_G5_roi3/val_actionness.pkl --n_layers 1 --pos_enc 0 --num_local 0 --local_type qkv --dilated_mask 0 --groupwise_heads 5 --roi_poolsize 3 --feat_model feature_anet_200
