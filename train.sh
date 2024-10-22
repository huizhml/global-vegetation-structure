#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small-g
#SBATCH --cpus-per-task=56
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
##SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu08fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch
#SBATCH --time=2-00:00:00
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
# export PATH="~/scratch/ffcv/bin:$PATH"

if [[ ${HOME} == "/users*" ]]; then
    echo "Running on LUMI"
    export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
    export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
    module load CrayEnv LUMI/23.09  partition/G
    module load rocm/5.4.6
    if [ $SLURM_LOCALID -eq 0 ] ; then
        rm -rf $MIOPEN_USER_DB_PATH
        mkdir -p $MIOPEN_USER_DB_PATH
    fi
else
    echo "Running on Hendrix"
    conda activate ffcv
fi

hostname

echo using training subsets and multiple gpus, ffcv.;
python run.py fit -c config/train.yaml  --model config/model/standard_unet.yaml \
        --data.init_args.train_fp ${HOME}/data/GEDI/train_subsets/train0_attrs_filtered.beton \
        --data.init_args.val_fp ${HOME}/data/GEDI/train_subsets/train1_attrs_filtered.beton \
        --data.init_args.batch_size 100 --data.init_args.num_workers 2 \
        --data.init_args.batches_ahead 3


# echo using training subsets and multiple gpus, h5.;
# python run.py fit -c config/train.yaml --data.init_args.train_fp ~/flash/data/train_h5s \
#     --data.init_args.batch_size 16384 --data.init_args.batches_ahead 3 

echo using MPIterDataset, single GPU, h5.;
python run.py fit -c config/train.yaml --data config/data/mp_iter.yaml \
    --data.init_args.batch_size 16384 --data.init_args.batches_ahead 3 \
    --data.init_args.num_jobs 56