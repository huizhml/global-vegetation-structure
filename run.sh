#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small-g
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
##SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu08fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
hostname

# export HYDRA_FULL_ERROR=1

cleanup() {
    rm /tmp/merged_h5_test.h5
    exit 0
}

trap 'cleanup' SIGTERM

mkdir /tmp/GEDI
time rsync --sparse=always -r ~/data/merged_h5_test.h5 /tmp/
module load LUMI/23.09 PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240208
# singularity exec $SIF bash -c '$WITH_CONDA ;
#             python run.py fit -c config/train.yaml \
#                 --optimizer.class_path torch.optim.SGD \
#                 --optimizer.init_args.lr 0.0001 \
#                 --optimizer.init_args.momentum 0.9 \
#                 --trainer.logger.init_args.name model_delta_rh
#             '
singularity exec $SIF bash -c '$WITH_CONDA ;
            python run.py fit -c config/train.yaml \
                --optimizer.class_path torch.optim.SGD \
                --optimizer.init_args.lr 0.00001 \
                --optimizer.init_args.momentum 0.9 \
                --optimizer.init_args.weight_decay 0.01 \
                --model.init_args.last_activation torch.nn.Sequential \
                --trainer.logger.init_args.name model_rh
            '
# singularity exec $SIF /runscripts/conda-python-simple \
#     -c 'import torch; print("I have this many devices:", torch.cuda.device_count())'
# python run.py fit -c config/train.yaml \
                # --model.init_args.last_activation torch.nn.Sequential
                # --trainer.logger.init_args.name model_rh