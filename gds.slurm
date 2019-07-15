#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=k80
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

# cd broi1d_pooling_avg
# python build.py
# cd ..

# cd broi1d_align
# python build.py
# cd ..

nvidia-smi

CUDA_VISIBLE_DEVICES=0 python binary_train.py thumos14 -b 64 --epochs 10 --n_layers 1 --pos_enc 0 --result_path ../ActivityNet/result/i3d_trained_tmp/seed1 --num_local 0 --local_type qkv --dilated_mask 0 --seed 1 --groupwise_heads 5 --roi_poolsize 3 --feat_model i3d_rgb_trained --use_flow 1 --only_flow 0 && \
python binary_test.py activitynet1.3 validation ../ActivityNet/result/i3d_trained_tmp/seed1/self_attn_N1_nopos_G5_roi3/model_best.pth.tar  ../ActivityNet/result/i3d_trained_tmp/seed1/self_attn_N1_nopos_G5_roi3/val_actionness.pkl --n_layers 1 --pos_enc 0 --num_local 0 --local_type qkv --dilated_mask 0 --groupwise_heads 5 --roi_poolsize 3 --feat_model i3d_rgb_trained --use_flow 1 --only_flow 0 && \
python gen_bottom_up_proposals.py ../ActivityNet/result/i3d_trained_tmp/seed1/self_attn_N1_nopos_G5_roi3/val_actionness.pkl --dataset activitynet --subset validation  --write_proposals data/activitynet1.3_tag_val_proposal_list.txt  --frame_path /data1/matheguo/important/data/activitynet/activity_net_frames