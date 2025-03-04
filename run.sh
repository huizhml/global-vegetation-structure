#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=gpu
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=16
##SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --time=1-23:50:00
#SBATCH --job-name=run
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu08fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch
echo "***************************** JOB INFO *****************************"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Number of tasks: $SLURM_NTASKS"
echo "Time requested: $SLURM_TIMELIMIT"
scontrol show job $SLURM_JOB_ID | grep "TRES="
ulimit -n 10000
echo 'ulimit' $(ulimit -n)
ulimit -n
echo "********************************************************************"

# CUDA_VISIBLE_DEVICES=0
train_data_name=train*_filtered_v1
val_data_name=val_filtered_v1_10m
debug=${2:-False}
if [ "$debug" = "True" ]; then
    train_data_name=debug*_filtered_v0
    val_data_name=debug0_filtered_v0
fi

# GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)
# effective_batch_size=8192
# BATCH_SIZE=$(($effective_batch_size / $GPU_COUNT))
# echo "Effective batch size: $effective_batch_size"
# echo "Batch size per GPU: $BATCH_SIZE"

sync_data_to_scratch() {
        echo "***************************** NODE INFO *****************************"
        # Check if hostname is in the list 
        local hostname=$(hostname)
        local number=$(echo "$hostname" | grep -o '[0-9]\+')
        echo "Number is $number"
        data_dir=${HOME}/data/GVS/train_subsets
        if hostname | grep -q "hendrix"; then
        host_list=("01" "02" "07" "22" "23" "24" "25" "26")
                if [[ " ${host_list[@]} " =~ " $number " ]]; then
                        echo "Host ${hostname} has an accessible scratch folder. Syncing data to /scratch."
                        # sync data to /scratch
                        mkdir -p /scratch/train_subsets
                        SOURCE_DIR="$HOME/data/GVS/train_subsets"
                        DEST_DIR="/scratch/train_subsets/"
                        for file in "$SOURCE_DIR"/train*_v1.beton; do
                                [ -f "$file" ] && rsync -av --progress "$file" "$DEST_DIR" &
                        done
                        #     rsync -av --progress ~/data/GEDI/train_subsets/${train_data_name}.beton /scratch/train_subsets/
                        rsync -av --progress ~/data/GVS/train_subsets/${val_data_name}.beton /scratch/train_subsets/ &
                        wait
                        rsync -av --progress ~/data/GVS/train_subsets/${val_data_name}.parquet /scratch/train_subsets/
                        data_dir=/scratch/train_subsets
                else
                        echo "Host ${hostname} doesn't have scratch folder. Skipping data sync."
                fi
        else
                
                echo 'data dir' $data_dir
                export WANDB_CACHE_DIR=${HOME}/scratch/wandb_cache

                ## Make sure GPUs are up
                if [ $SLURM_LOCALID -eq 0 ] ; then
                rocm-smi
                fi
                sleep 2

                # Set interfaces to be used by RCCL.
                # export PL_TORCH_DISTRIBUTED_BACKEND=gloo
                export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
                export NCCL_NET_GDR_LEVEL=3
                # export NCCL_DEBUG=INFO
                export NCCL_P2P_LEVEL=NVL
                export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
                export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
                export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
                # export HSA_ENABLE_SDMA=0
                # export HSA_ENABLE_SDMA_KERNEL=0
                export HSA_FORCE_FINE_GRAIN_PCIE=1
                amd_iommu=on
                iommu=pt
                ## Set MIOpen cache to a temporary folder.
                if [ $SLURM_LOCALID -eq 0 ] ; then
                rm -rf $MIOPEN_USER_DB_PATH
                mkdir -p $MIOPEN_USER_DB_PATH
                fi
                sleep 2

                # Report affinity
                echo "Rank $SLURM_PROCID --> $(taskset -p $$)"
        
        fi
        echo "********************************************************************"
}

id=$1
run_id=${run_id:-null}
max_epochs=${max_epochs:-200}
subcommand=${subcommand:-fit}
case $id in

# ====================================================================================
# Evaluation and prediction
# ====================================================================================
50)
echo predict tiles # need 128GB memory
# TODO: remove --model config/model/xception_s2.yaml
tile_id=32MQE
echo run prediction for model $run_id for tile $tile_id;
python run.py predict -c config/predict.yaml --model config/model/xception_s2.yaml \
        --data.init_args.input_lat_lon False \
        --data.init_args.num_workers 8 \
        --data.init_args.tile_id $tile_id \
        --correct_bias False \
        --trainer.logger.init_args.id $run_id

;;
51) 
val_data_name=val_filtered_v1
sync_data_to_scratch
echo run sparse evaluation for model $run_id on $val_data_name;
# TODO: remove --model config/model/xception_s2.yaml
python run.py test -c config/train.yaml --model config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.test_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 4096 \
        --data.init_args.distributed False \
        --model.init_args.evaluate_high_slope True \
        --correct_bias False \
        --bias_correction_column me_gradual_slope \
        --trainer.logger.init_args.id $run_id \
        --trainer.callbacks+=callbacks.prediction_logger.PredictionLogger
        ;;

52) 
sync_data_to_scratch
echo validate on run $run_id
val_data_name=val_filtered_v1
python run.py validate -c config/train.yaml \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 4096 \
        --data.init_args.distributed False \
        --trainer.logger.init_args.id $run_id
        ;;

# ====================================================================================
# ablation on mask and multi-task
# ====================================================================================

40)
sync_data_to_scratch
echo use latlon as an input
python run.py $subcommand -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.filter_out_nonveg True \
        --model.init_args.in_channels 15 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg_geo_lc
;;
41)
sync_data_to_scratch
echo quantile regression and land cover mapping;
python run.py fit -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.filter_out_nonveg True \
        --model.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg_lc
        ;;
42)
sync_data_to_scratch
# resume --trainer.logger.init_args.id 
echo quantile regression with latlon as input;
python run.py fit -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.filter_out_nonveg True \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.in_channels 15 \
        --model.init_args.feed_latlon True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg_geo
        ;;
43)
sync_data_to_scratch
echo quantile regression, zero out RH profile for building etc.;
python run.py fit -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.filter_out_nonveg True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg
;;
44)
sync_data_to_scratch
echo quantile regression, zero out RH profile for building etc.;
python run.py fit -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.filter_out_nonveg False \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR
;;

45)
sync_data_to_scratch
echo quantile regression, zero out RH profile for building etc.;
python run.py fit -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.zero_out_nonveg True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_zero_out
;;
46)
sync_data_to_scratch
# resume --trainer.logger.init_args.id 
echo quantile regression with latlon as input;
python run.py fit -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.zero_out_nonveg True \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.in_channels 15 \
        --model.init_args.feed_latlon True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_zero_out_geo
        ;;


47)
sync_data_to_scratch
echo quantile regression with latlon as input;
python run.py fit -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.zero_out_nonveg False \
        --model.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.in_channels 15 \
        --model.init_args.feed_latlon True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_geo
        ;;
48)
sync_data_to_scratch
echo quantile regression and land cover mapping;
python run.py fit -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.zero_out_nonveg True \
        --model.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_zero_out_lc
        ;;
49)
sync_data_to_scratch
echo quantile regression and land cover mapping;
python run.py fit -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.zero_out_nonveg False \
        --model.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_lc
        ;;
410)
sync_data_to_scratch
echo use latlon as an input
python run.py $subcommand -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.filter_out_nonveg False \
        --model.init_args.in_channels 15 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_geo_lc
;;

# ====================================================================================
# Loss
# ====================================================================================
30)
echo use shiftloss
sync_data_to_scratch
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.encoder.init_args.in_channels 15 \
        --model.init_args.out_channels 315 \
        --model.init_args.loss_fc.init_args.radius 1 \
        --model.init_args.zero_out_nonveg True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_and_LCC_zero_out_latlon_shiftloss
        ;;
31)
sync_data_to_scratch
echo using weighted loss;
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.encoder.init_args.in_channels 15 \
        --model.init_args.out_channels 315 \
        --model.init_args.encoder.init_args.scale_factor 2 \
        --model.init_args.loss_fc.init_args.weight_factor 0.003 \
        --model.init_args.zero_out_nonveg True \
        --lr_scheduler.init_args.num_warmup_steps 3490 \
        --trainer.max_epochs 400 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_and_LCC_zero_out_latlon_weigthed_loss
;;
# ====================================================================================
# Model architecture
# ====================================================================================
20)
t0=558400 # 400epochs, 1cycle, 418800 global steps for 300 epochs
num_sepconv_filters=256
max_lr=0.001
sync_data_to_scratch
echo "$data_dir/$val_data_name.beton"
python run.py fit -c config/train.yaml --model config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.restrict_rf False \
        --model.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.num_sepconv_blocks 8 \
        --model.init_args.activation_layer torch.nn.ReLU \
        --lr_scheduler.class_path models._scheduler.ChainedScheduler \
        --lr_scheduler.init_args.warmup_steps 6980 \
        --lr_scheduler.init_args.T_0 $t0 \
        --lr_scheduler.init_args.eta_min 0.000001 \
        --lr_scheduler.init_args.max_lr $max_lr \
        --lr_scheduler.init_args.gamma 0.2 \
        --use_pretrained_model True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name Xception_QR_eth # Xception_QR_rf_15_${num_sepconv_filters}_filters_3+${num_nonlin_blocks}blocks
        ;;
21)
t0=558400 # 400epochs, 1cycle, 418800 global steps for 300 epochs
num_sepconv_filters=512
max_lr=0.001
sync_data_to_scratch
echo "$data_dir/$val_data_name.beton"
ls "$data_dir/$val_data_name.beton"
echo Using Xception;
python run.py fit -c config/train.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.num_sepconv_blocks 3 \
        --model.init_args.nonlinear_order $nonlinear_order \
        --lr_scheduler.class_path models._scheduler.ChainedScheduler \
        --lr_scheduler.init_args.warmup_steps 6980 \
        --lr_scheduler.init_args.T_0 $t0 \
        --lr_scheduler.init_args.eta_min 0.000001 \
        --lr_scheduler.init_args.max_lr $max_lr \
        --lr_scheduler.init_args.gamma 0.2 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name Xception_QR_nonlin_${nonlinear_order} # Xception_QR_rf_15_${num_sepconv_filters}_filters_3+${num_nonlin_blocks}blocks
        ;;
22)
t0=558400 # 400epochs, 1cycle, 418800 global steps for 300 epochs
num_nonlin_blocks=4
num_sepconv_filters=512
max_lr=0.001
sync_data_to_scratch
echo "$data_dir/$val_data_name.beton"
ls "$data_dir/$val_data_name.beton"
echo Using Xception;
python run.py fit -c config/train.yaml --model config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.num_nonlin_blocks $num_nonlin_blocks \
        --model.init_args.activation_layer torch.nn.ReLU \
        --model.init_args.mid_block models.modules.xception_blocks.SepConvBlockSkip \
        --model.init_args.mlp_skip True \
        --lr_scheduler.class_path models._scheduler.ChainedScheduler \
        --lr_scheduler.init_args.warmup_steps 6980 \
        --lr_scheduler.init_args.T_0 $t0 \
        --lr_scheduler.init_args.eta_min 0.000001 \
        --lr_scheduler.init_args.max_lr $max_lr \
        --lr_scheduler.init_args.gamma 0.2 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name Xception_QR_${num_sepconv_filters}_filters_3+4_3x3_ReLU_add_middle_and_mlp_skip # Xception_QR_rf_15_${num_sepconv_filters}_filters_3+${num_nonlin_blocks}blocks
        ;;


# ====================================================================================
# Optimizer and scheduler
# ====================================================================================
10)
t0=558400 # 400epochs, 1cycle, 418800 global steps for 300 epochs
num_nonlin_blocks=4
num_sepconv_filters=256
max_lr=0.001
sync_data_to_scratch
echo "$data_dir/$val_data_name.beton"
ls "$data_dir/$val_data_name.beton"
echo Using Xception;
python run.py fit -c config/train.yaml --model config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.num_nonlin_blocks $num_nonlin_blocks \
        --model.init_args.activation_layer torch.nn.ReLU \
        --model.init_args.mid_block models.modules.xception_blocks.SepConvBlockSkip \
        --lr_scheduler.class_path models._scheduler.ChainedScheduler \
        --lr_scheduler.init_args.warmup_steps 6980 \
        --lr_scheduler.init_args.T_0 $t0 \
        --lr_scheduler.init_args.eta_min 0.000001 \
        --lr_scheduler.init_args.max_lr $max_lr \
        --lr_scheduler.init_args.gamma 0.2 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name Xception_QR_256_filters_3+4_3x3_ReLU_add_middle_skip # Xception_QR_rf_15_${num_sepconv_filters}_filters_3+${num_nonlin_blocks}blocks
        ;;
11)
# train_data_name=debug0_filtered_v1
# val_data_name=debug1_filtered_v1
# train_data_name=train9_filtered_v1
t0=558400 # 400epochs, 1cycle, 418800 global steps for 300 epochs
num_nonlin_blocks=3
num_sepconv_filters=512
max_lr=$SLURM_ARRAY_TASK_ID
max_lr=$(echo "scale=3; $max_lr / 1000.0" | bc)
echo "$max_lr"
sync_data_to_scratch
echo "$data_dir/$val_data_name.beton"
ls "$data_dir/$val_data_name.beton"
echo Using Xception;
python run.py fit -c config/train.yaml --model config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.num_nonlin_blocks $num_nonlin_blocks \
        --model.init_args.activation_layer torch.nn.ReLU \
        --lr_scheduler.class_path models._scheduler.ChainedScheduler \
        --lr_scheduler.init_args.warmup_steps 6980 \
        --lr_scheduler.init_args.T_0 $t0 \
        --lr_scheduler.init_args.eta_min 0.000001 \
        --lr_scheduler.init_args.max_lr $max_lr \
        --lr_scheduler.init_args.gamma 0.2 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name Xception_QR_512_filters_1cycle_maxlr${max_lr}_ReLU # Xception_QR_rf_15_${num_sepconv_filters}_filters_3+${num_nonlin_blocks}blocks
        ;;
12)
# train_data_name=debug*_filtered_v1
# val_data_name=debug0_filtered_v1
# train_data_name=train0_filtered_v1
# val_data_name=val_filtered_v1_1m
t0=558400 # 400epochs, 1cycle, 418800 global steps for 300 epochs
num_nonlin_blocks=3
num_sepconv_filters=512
max_lr=$SLURM_ARRAY_TASK_ID
max_lr=$(echo "scale=3; $max_lr / 1000.0" | bc)
echo "$max_lr"
sync_data_to_scratch
echo "$data_dir/$val_data_name.beton"
ls "$data_dir/$val_data_name.beton"
echo Using Xception;
python run.py fit -c config/train.yaml --model config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.num_nonlin_blocks $num_nonlin_blocks \
        --lr_scheduler.class_path models._scheduler.ChainedScheduler \
        --lr_scheduler.init_args.warmup_steps 6980 \
        --lr_scheduler.init_args.T_0 $t0 \
        --lr_scheduler.init_args.eta_min 0.000001 \
        --lr_scheduler.init_args.max_lr $max_lr \
        --lr_scheduler.init_args.gamma 0.2 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name Xception_QR_512_filters_1cycle_maxlr${max_lr} # Xception_QR_rf_15_${num_sepconv_filters}_filters_3+${num_nonlin_blocks}blocks
        ;;

13)
# train_data_name=val_filtered_v1_1m
# val_data_name=val_filtered_v1_1m
# train_data_name=train*_filtered_v1
# val_data_name=val_filtered_v1_1m
num_nonlin_blocks=3
num_sepconv_filters=256
sync_data_to_scratch
echo "$data_dir/$val_data_name.beton"
ls "$data_dir/$val_data_name.beton"
echo Using Xception;
python run.py fit -c config/train.yaml --model config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.num_nonlin_blocks $num_nonlin_blocks \
        --optimizer.init_args.lr 0.0006 \
        --lr_scheduler null \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name Xception_QR_lr_6e-4 # Xception_QR_rf_15_${num_sepconv_filters}_filters_3+${num_nonlin_blocks}blocks
        ;;
                # --trainer.callbacks+=callbacks.classification_logger.ClassificationLogger \
        # --trainer.callbacks.log_val_every=10 \
14) 
sync_data_to_scratch
echo Compare networks with different width; # sbatch --array=2,4 run.sh 14
    scale_factor=$SLURM_ARRAY_TASK_ID
    echo "$scale_factor"
    if [ "$scale_factor" -eq 4 ]; then
        lr=0.0005
    else
        lr=0.001
    fi

    python run.py fit -c config/train.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 1024 \
        --model.init_args.encoder.init_args.scale_factor $scale_factor \
        --model.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.zero_out_nonveg True \
        --model.init_args.encoder.init_args.in_channels 15 \
        --optimizer.init_args.lr $lr \
        --lr_scheduler.init_args.num_warmup_steps 6980 \
        --trainer.num_nodes 2 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name test_warmup_steps_10_epochs
;;

# sync_data_to_scratch
# echo find best weight decay;
# wd=$(echo "scale=6; $SLURM_ARRAY_TASK_ID / 1000000.0" | bc)
# wd=$(printf "%.6f" "$wd")
# echo "$wd" 
# echo debug, quantile regression, zero out RH profile for building etc.;
# python run.py fit -c config/train.yaml \
#         --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
#         --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
#         --data.init_args.batch_size 1024 \
#         --model.init_args.out_channels 315 \
#         --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
#         --model.init_args.feed_latlon True \
#         --model.init_args.zero_out_nonveg False \
#         --model.init_args.encoder.init_args.in_channels 15 \
#         --optimizer.init_args.weight_decay $wd \
#         --trainer.max_epochs $max_epochs \
#         --trainer.logger.init_args.name test_wd_$wd


# sync_data_to_scratch
#     echo find best inital learning rate;
#     lr=$(echo "scale=5; $SLURM_ARRAY_TASK_ID / 1000.0" | bc)
#     lr=$(printf "%.5f" "$lr")
#     echo "$lr" 

# python run.py fit -c config/train.yaml \
#         --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
#         --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
#         --data.init_args.batch_size 1024 \
#         --model.init_args.out_channels 315 \
#         --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
#         --model.init_args.feed_latlon True \
#         --model.init_args.zero_out_nonveg False \
#         --model.init_args.encoder.init_args.in_channels 15 \
#         --trainer.max_epochs $max_epochs \
#         --optimizer.init_args.lr $lr \
#         --trainer.logger.init_args.id $run_id \
#         --trainer.logger.init_args.name test_lr_$lr


*)
echo runnning nothing ;;
esac

echo finished