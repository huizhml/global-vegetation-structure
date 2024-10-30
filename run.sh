#!/bin/bash
##SBATCH --account=project_465000894
#SBATCH --partition=gpu
##SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --exclude hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu18fl #for using /scratch
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
hostname
conda activate ffcv

# sync data to /scratch
# mkdir -p /scratch/train_subsets
# rsync -av --progress ~/data/GEDI/train_subsets/train0_attrs_filtered.beton /scratch/train_subsets
# rsync -av --progress ~/data/GEDI/train_subsets/train1_attrs_filtered.beton /scratch/train_subsets/train1_attrs_filtered.beton &
# # rsync -av ~/data/GEDI/train_subsets/train*.beton /scratch/train_subsets &
# echo syncing data to /scratch
data_dir=${HOME}/data/GEDI/train_subsets #/scratch/train_subsets


id=$1
echo Running job $id
case $id in
1)
echo debuging, using debug1.beton;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/debug0_filtered.beton \
        --data.init_args.val_fp $data_dir/debug0_filtered.beton;;
2)
echo training, using beton subsets.;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/train1_filtered.beton \
        --data.init_args.val_fp $data_dir/train7_filtered.beton \
        --trainer.logger.init_args.name L1_loss;;
3)
echo training on beton subsets.;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/train1_filtered.beton \
        --data.init_args.val_fp $data_dir/train7_filtered.beton \
        --model.init_args.loss_fc.init_args.name mse \
        --trainer.logger.init_args.name L2_loss;;
4)
echo training on beton subsets. qualtile loss;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/train1_filtered.beton \
        --data.init_args.val_fp $data_dir/train7_filtered.beton \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --trainer.logger.init_args.name Quantile_loss;;
5)
echo find best inital learning rate;
lr=$(echo "scale=5; $SLURM_ARRAY_TASK_ID / 10000.0" | bc)
lr=$(printf "%.5f" "$lr")
echo "$lr" 

python run.py fit -c config/train.yaml \
        --optimizer.init_args.lr $lr \
        --data.init_args.train_fp $data_dir/train1_filtered.beton \
        --data.init_args.val_fp $data_dir/train7_filtered.beton \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --trainer.logger.init_args.name Quantile_loss;;
6)
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
;;
*)
echo runnning nothing ;;
esac

echo finished