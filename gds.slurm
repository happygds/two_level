#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=k40
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mail-type=end
#SBATCH --mail-user=YOUR@MAIL
#SBATCH --output=out.out
#SBATCH --error=err.err
#SBATCH --time=32:15:00

source /usr/share/Modules/init/bash
module purge
module load gcc/4.9 miniconda3 cuda/8.0
source activate py36

cd broi1d_pooling_avg
python build.py
cd ..

cd broi1d_align
python build.py
cd ..

nvidia-smi

python binary_train.py thumos14 -b 64 --epochs 10 --n_layers 1 --pos_enc 0 --result_path ../THUMOS14/result_two_level2/seed1 --num_local 0 --local_type qkv --dilated_mask 0 --seed 1 --groupwise_heads 5 --roi_poolsize 3 --feat_model feature_anet_200 --resume ../THUMOS14/result_two_level2/seed1/self_attn_N1_nopos_G5_roi3/model_best.pth.tar && \
python binary_test.py thumos14 testing ../THUMOS14/result_two_level2/seed1/self_attn_N1_nopos_G5_roi3/model_best.pth.tar  ../THUMOS14/result_two_level2/seed1/self_attn_N1_nopos_G5_roi3/test_actionness.pkl --n_layers 1 --pos_enc 0 --num_local 0 --local_type qkv --dilated_mask 0 --groupwise_heads 5 --roi_poolsize 3 --feat_model feature_anet_200 && \
python gen_bottom_up_proposals.py ../THUMOS14/result_two_level2/seed1/self_attn_N1_nopos_G5_roi3/test_actionness.pkl --dataset thumos14 --subset testing  --write_proposals data/thumos14_tag_val_proposal_list.txt  --frame_path ../THUMOS14/frames