#!/bin/bash
##SBATCH --account=project_465000894
#SBATCH --partition=gpu
##SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --exclude hendrixgpu03fl,hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl #for using /scratch
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk

train_data_name=train1_filtered_v1.beton
val_data_name=train7_filtered_v1.beton
debug=${2:-False}
if [ "$debug" = "True" ]; then
    train_data_name=debug0_filtered_v1.beton
    val_data_name=debug0_filtered_v1.beton
fi

# Check if hostname is in the list
hostname=$(hostname)
number=$(echo "$hostname" | grep -o '[0-9]\+')
echo "Number is $number"
host_list=("01" "02" "07" "16" "22")
if [[ " ${host_list[@]} " =~ " $number " ]]; then
    echo "Host ${hostname} has an accessible scratch folder. Syncing data to /scratch."
    # sync data to /scratch
    mkdir -p /scratch/train_subsets
    rsync -av --progress ~/data/GEDI/train_subsets/$train_data_name /scratch/train_subsets/
    rsync -av --progress ~/data/GEDI/train_subsets/$val_data_name /scratch/train_subsets/
    data_dir=/scratch/train_subsets
else
    echo "Host ${hostname} doesn't have scratch folder. Skipping data sync."
    data_dir=${HOME}/data/GEDI/train_subsets
fi


id=$1
echo Running job $id
case $id in
1)
echo debuging, using debug1.beton;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/$train_data_name \
        --data.init_args.val_fp $data_dir/$val_data_name \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.loss_fc.zero_out True \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --trainer.callbacks+=callbacks.classification_logger.ClassificationLogger \
        --trainer.callbacks.log_val_every=1
        ;;
2)
echo training, using beton subsets.;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/$train_data_name \
        --data.init_args.val_fp $data_dir/$val_data_name \
        --trainer.logger.init_args.name L1_loss;;
3)
echo training on beton subsets.;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/$train_data_name \
        --data.init_args.val_fp $data_dir/$val_data_name \
        --model.init_args.loss_fc.init_args.name mse \
        --trainer.logger.init_args.name L2_loss;;
4)
# resume --trainer.logger.init_args.id 
echo training on beton subsets. qualtile loss; 
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/$train_data_name \
        --data.init_args.val_fp $data_dir/$val_data_name \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --trainer.max_epochs 200 \
        --trainer.logger.init_args.id 4fd8j93r \
        --trainer.logger.init_args.name Quantile_loss
        ;;
5)
    echo find best inital learning rate;
    lr=$(echo "scale=5; $SLURM_ARRAY_TASK_ID / 10000.0" | bc)
    lr=$(printf "%.5f" "$lr")
    echo "$lr" 

    python run.py fit -c config/train.yaml \
            --optimizer.init_args.lr $lr \
            --data.init_args.train_fp $data_dir/$train_data_name \
            --data.init_args.val_fp $data_dir/$val_data_name \
            --model.init_args.out_channels 303 \
            --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
            --trainer.logger.init_args.name Quantile_loss;;
6)
    echo sanity check for quantile regression;
    data_dir=${HOME}/data/GEDI/train_subsets
    python run.py fit -c config/train.yaml \
            --data.init_args.train_fp $data_dir/debug1_filtered.beton \
            --data.init_args.val_fp $data_dir/debug1_filtered.beton \
            --data.init_args.batch_size 100 \
            --lr_scheduler.init_args.step_size 100 \
            --model.init_args.out_channels 101 \
            --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
            --model.init_args.loss_fc.init_args.quantiles [0.5] \
            --trainer.max_epochs 1000 \
            --trainer.logger.init_args.name Sanity_check_Quantile_loss_median;

    python run.py fit -c config/train.yaml \
            --data.init_args.train_fp $data_dir/debug1_filtered.beton \
            --data.init_args.val_fp $data_dir/debug1_filtered.beton \
            --data.init_args.batch_size 100 \
            --model.init_args.out_channels 101 \
            --lr_scheduler.init_args.step_size 100 \
            --trainer.max_epochs 1000 \
            --trainer.logger.init_args.name Sanity_check_L1_loss;;
7)
echo Quantile regression and land cover mapping;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/$train_data_name \
        --data.init_args.val_fp $data_dir/$val_data_name \
        --model.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --trainer.callbacks+=callbacks.classification_logger.ClassificationLogger \
        --trainer.callbacks.log_val_every=10 \
        --trainer.max_epochs 200 \
        --trainer.logger.init_args.id allse0dy \
        --trainer.logger.init_args.name Quantile_CE_loss
        ;;
8)
echo quantile regression, zero out RH profile for building etc.;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp $data_dir/$train_data_name \
        --data.init_args.val_fp $data_dir/$train_data_name \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.loss_fc.zero_out True \
        --trainer.max_epochs 200 \
        --trainer.logger.init_args.id 05lp279m \
        --trainer.logger.init_args.name Quantile_loss_zero_out
;;
*)
echo runnning nothing ;;
esac

echo finished