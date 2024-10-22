#!/bin/bash
##SBATCH --account=project_465000894
#SBATCH --partition=gpu
##SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu08fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
hostname
conda activate ffcv

# sync data to /scratch
mkdir -p /scratch/train_subsets
rsync -av --progress ~/data/GEDI/train_subsets/train0.beton /scratch/train_subsets
rsync -av --progress ~/data/GEDI/train_subsets/val.beton /scratch/train_subsets/val.beton &
# rsync -av ~/data/GEDI/train_subsets/train*.beton /scratch/train_subsets &
echo syncing data to /scratch
data_dir=/scratch/train_subsets


id=$SLURM_ARRAY_TASK_ID
echo SLURM_ARRAY_TASK_ID $id
case $id in
1)
echo debuging, using debug1.beton;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp  $data_dir/debug1.beton \
        --data.init_args.val_fp  $data_dir/debug1.beton \
        --data.init_args.batch_size 100 \
        --data.init_args.num_workers 4 \
        --model.init_args.loss models.losses.robust_loss.AdaptiveLossFunction \
        --model.init_args.loss.init_args.num_dims 101;;
2)
echo running job 2 ;
echo training, using beton subsets.;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/train0.beton \
        --data.init_args.val_fp $data_dir/val.beton \
        --trainer.logger.init_args.name L1_loss;;
3)
echo running job 3 ;
echo training, using beton subsets.;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/train0.beton \
        --data.init_args.val_fp $data_dir/val.beton \
        --model.init_args.loss.init_args.name mse \
        --trainer.logger.init_args.name L2_loss;;
4)
echo running job 4 ;
train_fp=$1
if ["$train_fp" == "~/flash/data/debug1.beton"]; then
    echo debug AdaptiveRobustLoss;
    python run.py fit -c config/train.yaml --data.init_args.train_fp $train_fp \
        --data.init_args.val_fp $train_fp \
        --data.init_args.batch_size 100 \
        --model.init_args.loss models.losses.robust_loss.AdaptiveLossFunction \
        --model.init_args.loss.init_args.num_dims 101
else
    echo train using AdaptiveRobustLoss;
    python run.py fit -c config/train.yaml --data.init_args.train_fp ~/flash/data/train_h5s \
        --model.init_args.loss models.losses.robust_loss.RobustLoss \
        --model.init_args.loss.init_args.num_dms 101
fi


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
6) 
echo running job 6;
python run.py fit -c config/train.yaml ;;
*)
echo runnning nothing ;;
esac

echo finished