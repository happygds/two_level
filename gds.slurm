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
#SBATCH --time=06:15:00

source /usr/share/Modules/init/bash
module purge
module load gcc/4.9 cuda/8.0 cudnn/5.1 anaconda/3 opencv/3.1
source activate py36

python main.py --no_train 1 --no_val 1 --test 1 --convert_model 1 --test_subset all
