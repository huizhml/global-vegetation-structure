#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small-g
#SBATCH --cpus-per-task=32
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

# cleanup() {
#     rm /tmp/merged_h5_test.h5
#     exit 0
# }

# trap 'cleanup' SIGTERM
# Make sure GPUs are up
if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
fi
sleep 2

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

# Set ROCR_VISIBLE_DEVICES so that each task uses the proper GPU
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID

# Report affinity to check
echo "Rank $SLURM_PROCID --> $(taskset -p $$); GPU $ROCR_VISIBLE_DEVICES"

# The usual PyTorch initialisations (also needed on NVIDIA)
# Note that since we fix the port ID it is not possible to run, e.g., two
# instances via this script using half a node each.
export MASTER_ADDR=$(/runscripts/get-master "$SLURM_NODELIST")
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID

mkdir /tmp/GEDI
# time rsync --sparse=always -r ~/data/merged_h5_test.h5 /tmp/
module load LUMI/23.09 PyTorch/2.2.2-rocm-5.6.1-python-3.10-singularity-20240404
export NUMEXPR_MAX_THREADS=64

id=$SLURM_ARRAY_TASK_ID
echo SLURM_ARRAY_TASK_ID $id
case $id in
1)
echo running job 1 ;
singularity exec $SIF bash -c '$WITH_CONDA_VENV ;
            python run.py fit -c config/train.yaml \
                --data.use_subset True \
                 --optimizer.class_path adabelief_pytorch.AdaBelief \
                 --optimizer.init_args.lr 1e-4 --optimizer.weight_decay 0.06 \
                --trainer.logger.init_args.name model_delta_rh
            ';;
2)
echo running job 2 ;
singularity exec $SIF bash -c '$WITH_CONDA_VENV ;
            python run.py fit -c config/train.yaml -c config/train_model_rh.yaml \
                --optimizer.class_path adabelief_pytorch.AdaBelief \
                --optimizer.init_args.lr 1e-4 --optimizer.weight_decay 0.06 \
                --trainer.logger.init_args.name model_rh_mse_hinge \
                --model.init_args.loss models.losses.mse_hinge.MSEHinge
            ';;
3)
echo running job 3 ;
singularity exec $SIF bash -c '$WITH_CONDA_VENV ;
            python run.py fit -c config/train.yaml -c config/train_model_rh.yaml \
                --optimizer.class_path adabelief_pytorch.AdaBelief \
                --optimizer.init_args.lr 1e-4 --optimizer.weight_decay 0.06 \
                --trainer.logger.init_args.name model_rh_mse
            ';;
4)
echo running job 4 ;
singularity exec $SIF bash -c '$WITH_CONDA_VENV ;
            python run.py fit -c config/train.yaml -c config/train_model_rh.yaml \
                --optimizer.class_path adabelief_pytorch.AdaBelief \
                --optimizer.init_args.lr 1e-4 --optimizer.weight_decay 0.06 \
                --trainer.logger.init_args.name model_rh_GNL_homo \
                --model.init_args.loss models.losses.gaussian_nl.GNLLoss \
                --model.init_args.out_channels 102
            ';;
5)
echo running job 5 ;
singularity exec $SIF bash -c '$WITH_CONDA_VENV ;
            python run.py fit -c config/train.yaml -c config/train_model_rh.yaml \
                --optimizer.class_path adabelief_pytorch.AdaBelief \
                --optimizer.init_args.lr 1e-4 --optimizer.weight_decay 0.06 \
                --trainer.logger.init_args.name model_rh_GNL_hetero \
                --model.init_args.loss models.losses.gaussian_nl.GNLLoss \
                --model.init_args.out_channels 202
            ';;      
*)
echo runnning nothing ;;
esac

echo finished