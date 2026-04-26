#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:50:00
#SBATCH --job-name=run
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
#SBATCH --exclude hendrixgpu11fl,hendrixgpu12fl
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
        data_dir=${HOME}/data/gvs/train_subsets
        if hostname | grep -q "hendrix"; then
        host_list=("01" "02" "07" "22" "23" "24" "25" "26")
                if [[ " ${host_list[@]} " =~ " $number " ]]; then
                        echo "Host ${hostname} has an accessible scratch folder. Syncing data to /scratch."
                        # sync data to /scratch
                        mkdir -p /scratch/train_subsets
                        SOURCE_DIR="$HOME/data/gvs/train_subsets"
                        DEST_DIR="/scratch/train_subsets/"
                        for file in "$SOURCE_DIR"/train*_v1.beton; do
                                [ -f "$file" ] && rsync -av --progress "$file" "$DEST_DIR" &
                        done
                        #     rsync -av --progress ~/data/GEDI/train_subsets/${train_data_name}.beton /scratch/train_subsets/
                        rsync -av --progress ~/data/gvs/train_subsets/${val_data_name}.beton /scratch/train_subsets/ &
                        wait
                        rsync -av --progress ~/data/gvs/train_subsets/${val_data_name}.parquet /scratch/train_subsets/
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
80)
# ====================================================================================
# Conformal prediction
# ====================================================================================
# NOTE: takes about 1h5min on LUMI for val_filtered_v1
# NOTE: takes about 35min on A100 for val_filtered_v1
run_id=cg11fpjr
test_data_name=test_filtered_v1
# sync_data_to_scratch
data_dir=${HOME}/data/gvs/train_subsets
correct_bias=True
echo get sparse prediction for model $run_id on $test_data_name;
python run.py test -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.test_fp "$data_dir/$test_data_name.beton" \
        --data.init_args.batch_size 4096 \
        --data.init_args.distributed False \
        --model.init_args.evaluate_high_slope True \
        --correct_bias $correct_bias \
        --bias_correction_column me_gradual_slope_veg \
        --recalculate_bias False \
        --trainer.logger.init_args.id $run_id \
        --trainer.callbacks+=callbacks.prediction_logger.PredictionLogger \
        --trainer.callbacks.save_dir ~/data/gvs/uncertainty/ \
        --trainer.callbacks.outfile_suffix _corrected
        # --trainer.callbacks.output_rh_idxs "[0,25,50,95,98,100]"

;;

# ====================================================================================
# Quantization-aware training
# ====================================================================================
70)
# ******************************
# 1. Need multiple GPUs, I got CUDA out of memory error when using 1 GPU
# 2. Takes about 15min to train one epoch 7min to evaluate one epoch on 4 L40s

# ******************************
max_epochs=100
sync_data_to_scratch
echo "$data_dir/$val_data_name.beton"
ls "$data_dir/$val_data_name.beton"
echo QAT finetuning;
module load gcc/11.2.0
run_id=cg11fpjr
python run.py $subcommand -c config/qat.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.cal_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.distributed True \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.filter_out_nonveg True \
        --model.init_args.backbone.init_args.in_channels 15 \
        --quantize_model True \
        --correct_bias False \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg_geo_lc_QAT_finetune
        ;;
71)
echo Calibrate model for QAT
sync_data_to_scratch
echo "$data_dir/$val_data_name.beton"
ls "$data_dir/$val_data_name.beton"
echo Using Xception;
module load gcc/11.2.0
python run.py fit -c config/qat.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.distributed False \
        --data.init_args.batch_size 1024 \
        --data.init_args.order 'SEQUENTIAL' \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.filter_out_nonveg True \
        --model.init_args.backbone.init_args.in_channels 15 \
        --quantize_model True \
        --correct_bias False \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg_geo_lc_QAT_test
;;

72)
echo run onnx inference
tile_id=32MQE
year=2020
comp=None
level=7
python run.py predict -c config/deploy.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.input_lat_lon True \
        --data.init_args.num_workers 8 \
        --data.init_args.tile_id $tile_id \
        --data.init_args.pred_fp ~/data/gvs/deploy/inference_${year}.zarr \
        --data.init_args.prediction_dir ~/data/gvs/deploy/predictions_${year} \
        --correct_bias False \
        --data.init_args.comp_level $level \
        --data.init_args.compression $comp \
        --data.init_args.patch_size 544 \
        --data.init_args.chunk_size 512 \
        --data.init_args.output_format cog \
        --model.init_args.onnx_model_path checkpoints/fake_quantized_model_finetuned1_epochs_ycfb24ae.onnx \
        --trainer.logger.init_args.resume False \
        --trainer.logger.init_args.name QR_veg_geo_lc_QAT_inference_test
;;

# ====================================================================================
# Downstream tasks
# ====================================================================================
60)
# sync_data_to_scratch
run_id=cg11fpjr
echo $run_id
echo predict for downstream task;
python run.py predict -c config/predict.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.class_path datasets.h5_dataset.SparsePredDataModule \
        --data.init_args.pred_fp ~/data/gvs/downstream_tasks/naturalness/results_from_vsm_2017/s2_gedi_patches_ps31/s2_2017_ps31.h5 \
        --data.init_args.prediction_dir ~/data/gvs/downstream_tasks/naturalness/results_from_vsm_2017/vsm_patches_ps15_single_h5/ \
        --data.init_args.batch_size 2048 \
        --trainer.logger.init_args.id $run_id
;;
61)
# sync_data_to_scratch
run_id=0crmfaia
echo $run_id
echo downstream task training with full profile;
python run.py fit -c config/train_naturalness.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.class_path datasets.h5_dataset.NaturalnessDataModule \
        --data.init_args.h5_file ~/data/gvs/downstream_task_data/rhs_predictions_2017_${run_id}_ps31.h5 \
        --data.init_args.naturalness_fp ~/data/gvs/downstream_task_data/naturalness/reference_data_set_updated.with_images.csv \
        --data.init_args.use_full_profile True \
        --model.init_args.mean_std_fp ~/data/gvs/downstream_task_data/naturalness/mean_std_${run_id}.npz \
        --model.init_args.backbone.init_args.in_channels 113 \
        --model.init_args.backbone.init_args.out_channels 7 \
        --data.init_args.class_balance False \
        --trainer.logger.init_args.name naturalness_s2_rhs
;;

62)
# sync_data_to_scratch
run_id=0crmfaia
echo $run_id
echo downstream task training with s2 only;
python run.py fit -c config/train_naturalness.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.class_path datasets.h5_dataset.NaturalnessDataModule \
        --data.init_args.h5_file ~/data/gvs/downstream_task_data/rhs_predictions_2017_${run_id}_ps31.h5 \
        --data.init_args.naturalness_fp ~/data/gvs/downstream_task_data/naturalness/reference_data_set_updated.with_images.csv \
        --model.init_args.mean_std_fp ~/data/gvs/downstream_task_data/naturalness/mean_std_${run_id}.npz \
        --model.init_args.backbone.init_args.in_channels 12 \
        --model.init_args.backbone.init_args.out_channels 7 \
        --data.init_args.class_balance False \
        --trainer.logger.init_args.name naturalness_s2
;;

63)
# sync_data_to_scratch
run_id=0crmfaia
echo $run_id
echo downstream task training with s2 and top height;
python run.py fit -c config/train_naturalness.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.class_path datasets.h5_dataset.NaturalnessDataModule \
        --data.init_args.h5_file ~/data/gvs/downstream_task_data/rhs_predictions_2017_${run_id}_ps31.h5 \
        --data.init_args.naturalness_fp ~/data/gvs/downstream_task_data/naturalness/reference_data_set_updated.with_images.csv \
        --data.init_args.mean_std_fp ~/data/gvs/downstream_task_data/naturalness/mean_std_${run_id}.npz \
        --data.init_args.use_full_profile False \
        --model.init_args.transform.init_args.input_top_height True \
        --model.init_args.transform.init_args.mean_std_fp ~/data/gvs/downstream_task_data/naturalness/mean_std_${run_id}.npz \
        --model.init_args.backbone.init_args.in_channels 13 \
        --model.init_args.backbone.init_args.out_channels 7 \
        --data.init_args.class_balance False \
        --trainer.logger.init_args.name naturalness_s2_rh98
;;
64)
# sync_data_to_scratch
run_id=0crmfaia
echo $run_id
echo downstream task training with rhs only;
python run.py fit -c config/train_naturalness.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.class_path datasets.h5_dataset.NaturalnessDataModule \
        --data.init_args.h5_file ~/data/gvs/downstream_task_data/rhs_predictions_2017_${run_id}_ps31.h5 \
        --data.init_args.naturalness_fp ~/data/gvs/downstream_task_data/naturalness/reference_data_set_updated.with_images.csv \
        --data.init_args.use_full_profile True \
        --model.init_args.mean_std_fp ~/data/gvs/downstream_task_data/naturalness/mean_std_${run_id}.npz \
        --model.init_args.backbone.init_args.in_channels 101 \
        --model.init_args.backbone.init_args.out_channels 7 \
        --lr_scheduler.init_args.max_lr 0.0001 \
        --data.init_args.class_balance False \
        --trainer.logger.init_args.name naturalness_rhs_only
;;

65)
# sync_data_to_scratch
run_id=0crmfaia
echo $run_id
echo downstream task training with rhs only;
python run.py fit -c config/train_naturalness.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.class_path datasets.h5_dataset.NaturalnessDataModule \
        --data.init_args.h5_file ~/data/gvs/downstream_task_data/rhs_predictions_2017_${run_id}_ps31.h5 \
        --data.init_args.naturalness_fp ~/data/gvs/downstream_task_data/naturalness/reference_data_set_updated.with_images.csv \
        --data.init_args.use_full_profile False \
        --model.init_args.mean_std_fp ~/data/gvs/downstream_task_data/naturalness/mean_std_${run_id}.npz \
        --model.init_args.backbone.init_args.in_channels 1 \
        --model.init_args.backbone.init_args.out_channels 7 \
        --data.init_args.class_balance False \
        --trainer.logger.init_args.name naturalness_rh98
;;

66)
echo run random forest with s2 and rhs;
python run_rf.py
;;

67)
echo run random forest with s2 only;
python run_rf.py s2_only=True
;;

68)
echo run random forest with rhs only;
python run_rf.py rhs_only=True
;;

69)
echo run random forest with s2 and rh98;
python run_rf.py use_full_profile=False
;;

# ====================================================================================
# * Evaluation and prediction
# 1. Bias correction
#    - takes about 18min/epoch on LUMI, 40min on titanrtx (single GPU)
# ====================================================================================

50)
# ******************************
# 1. Need 48GB memory
# 2. Takes about 21 min to predict one tile on one L40s (write 2 bands)
# 3. Takes about 31 min to predict one tile on one L40s (write all 303 bands)
# ******************************
echo predict tiles on Hendrix
tile_id=21MTM # 32MQE
year=2020
run_id=cg11fpjr
echo run prediction for model $run_id for tile $tile_id in year $year;
python run.py predict -c config/predict.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.input_lat_lon True \
        --data.init_args.num_workers 4 \
        --data.init_args.tile_id $tile_id \
        --data.init_args.stream_input False \
        --data.init_args.metadata_file none  \
        --data.init_args.pred_fp ~/data/gvs/deploy/inference_${year}.zarr \
        --data.init_args.prediction_dir ~/data/gvs/deploy/predictions_GTiff_${year}_test/${tile_id}_GTiff \
        --data.init_args.year $year \
        --correct_bias True \
        --data.init_args.patch_size 544 \
        --data.init_args.chunk_size 512 \
        --data.init_args.debug False \
        --data.init_args.predict_full_profile True \
        --data.init_args.output_format gtiff \
        --trainer.logger.init_args.resume False \
        --trainer.logger.init_args.offline True \
        --trainer.logger.init_args.id $run_id 
;;
56)

echo debug tile prediction on Hendrix, saving intermediate tif
tile_id=20LQQ # 32MQE
year=2020
run_id=cg11fpjr
echo run prediction for model $run_id for tile $tile_id in year $year;
python run.py predict -c config/predict.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.input_lat_lon True \
        --data.init_args.num_workers 4 \
        --data.init_args.tile_id $tile_id \
        --data.init_args.stream_input False \
        --data.init_args.metadata_file none  \
        --data.init_args.pred_fp ~/data/gvs/deploy/inference_${year}.zarr \
        --data.init_args.prediction_dir ~/data/gvs/deploy/predictions_GTiff_${year}_test/${tile_id}_GTiff \
        --data.init_args.year $year \
        --correct_bias True \
        --data.init_args.patch_size 544 \
        --data.init_args.chunk_size 512 \
        --data.init_args.debug False \
        --data.init_args.save_intermediate_tif True \
        --data.init_args.predict_full_profile True \
        --data.init_args.output_format gtiff \
        --trainer.logger.init_args.resume False \
        --trainer.logger.init_args.offline True \
        --trainer.logger.init_args.id $run_id 
;;
54)
# ******************************
# 1. Need 48GB memory
# 2. Takes ~21 min to predict one tile on one L40s (write 2 bands)
# 3. Takes ~31 min to predict one tile on one L40s (write all 303 bands)
#/scratch/$tile_id
# ******************************
echo predict tiles on LUMI
tile_id=20LPP # 32MQE
year=2020
run_id=cg11fpjr
echo run prediction for model $run_id for tile $tile_id in year $year;
python run.py predict -c config/predict.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.tile_id $tile_id \
        --data.init_args.metadata_file ~/data/gvs/deploy/slurm_job_files_${year}/deploy_s2_items_${year}_part4.parquet \
        --data.init_args.download_data True \
        --data.init_args.pred_fp ~/flash/data/gvs/deploy/inference_${year} \
        --data.init_args.prediction_dir ~/flash/data/gvs/deploy/predictions_GTiff_${year}/${tile_id}_GTiff \
        --data.init_args.year $year \
        --data.init_args.cache_predictions False \
        --data.init_args.stream_input True \
        --data.init_args.debug False \
        --trainer.logger.init_args.resume False \
        --trainer.logger.init_args.offline False \
        --trainer.logger.init_args.save_dir /tmp \
        --trainer.logger.init_args.id $run_id 
;;
55)
# ******************************
# 1. Need 128GB memory
# 2. Takes ~21 min to predict one tile on one L40s (write 2 bands)
# 3. Takes ~31 min to predict one tile on one L40s (write all 303 bands)
#/scratch/$tile_id
# ******************************
echo predict tiles 
tile_id=32MQE # 32MQE
year=2020
run_id=cg11fpjr
echo run prediction for model $run_id for tile $tile_id;
python run.py predict -c config/predict.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.input_lat_lon True \
        --data.init_args.num_workers 4 \
        --data.init_args.tile_id $tile_id \
        --data.init_args.pred_fp ~/data/gvs/deploy/inference_${year}.zarr \
        --data.init_args.prediction_dir ~/data/gvs/deploy/predictions_${year}/${tile_id}_fp32_infer \
        --data.init_args.year $year \
        --data.init_args.batch_size 1 \
        --data.init_args.cache_predictions False \
        --correct_bias True \
        --data.init_args.comp_level 7 \
        --data.init_args.compression None \
        --data.init_args.patch_size 544 \
        --data.init_args.chunk_size 512 \
        --data.init_args.debug False \
        --data.init_args.predict_full_profile False \
        --data.init_args.output_format gtiff \
        --trainer.logger.init_args.resume False \
        --trainer.logger.init_args.id $run_id 
;;
51) 
# NOTE: takes about 1h5min on LUMI for val_filtered_v1
# NOTE: takes about 35min on A100 for val_filtered_v1
val_data_name=cal_filtered_v1
sync_data_to_scratch
correct_bias=True
echo get sparse prediction for model $run_id on $val_data_name;
python run.py test -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.test_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 4096 \
        --data.init_args.distributed False \
        --model.init_args.evaluate_high_slope True \
        --correct_bias $correct_bias \
        --bias_correction_column me_gradual_slope_veg \
        --recalculate_bias False \
        --trainer.logger.init_args.id $run_id \
        --trainer.callbacks+=callbacks.prediction_logger.PredictionLogger \
        --trainer.callbacks.output_rh_idxs "[0,25,50,95,98,100]"

echo generate the comparison table for canopy height predictions;
python evaluate.py run_id=$run_id corrected=$correct_bias
;;
52) 
val_data_name=val_filtered_v1
sync_data_to_scratch
echo Get boxplot data for top height and biome aggregated analysis on run $run_id
python run.py validate -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.batch_size 4096 \
        --data.init_args.distributed False \
        --model.init_args.evaluate_high_slope False \
        --correct_bias True \
        --bias_correction_column me_gradual_slope_veg \
        --trainer.logger.init_args.id $run_id \
        --trainer.callbacks+=callbacks.boxplot.BoxplotLogger
        ;;

53)
echo Test compression and comp_level
run_id=cg11fpjr
tile_id=32MQE
year=2017
compression=$comp
echo compression $compression
echo run prediction for model $run_id for tile $tile_id;
python run.py predict -c config/predict.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.input_lat_lon True \
        --data.init_args.num_workers 8 \
        --data.init_args.tile_id $tile_id \
        --data.init_args.pred_fp ~/data/gvs/deploy/inference_${year}.zarr \
        --data.init_args.prediction_dir ~/data/gvs/deploy/predictions_${year} \
        --data.init_args.comp_level $SLURM_ARRAY_TASK_ID \
        --data.init_args.compression $compression \
        --correct_bias True \
        --trainer.logger.init_args.offline True \
        --trainer.logger.init_args.log_model False \
        --trainer.logger.init_args.id $run_id
;;

# ====================================================================================
# ablation on mask and multi-task
# ====================================================================================

40)
sync_data_to_scratch
echo use latlon as an input
python run.py $subcommand -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.distributed False \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.filter_out_nonveg True \
        --model.init_args.backbone.init_args.in_channels 15 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg_geo_lc
;;
41)
sync_data_to_scratch
echo quantile regression and land cover mapping;
python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.filter_out_nonveg True \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg_lc
        ;;
42)
sync_data_to_scratch
# resume --trainer.logger.init_args.id 
echo quantile regression with latlon as input;
python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.filter_out_nonveg True \
        --model.init_args.backbone.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.backbone.init_args.in_channels 15 \
        --model.init_args.feed_latlon True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg_geo
        ;;
43)
sync_data_to_scratch
echo quantile regression, zero out RH profile for building etc.;
python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.filter_out_nonveg True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg
;;
44)
sync_data_to_scratch
echo quantile regression, zero out RH profile for building etc.;
python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.filter_out_nonveg False \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR
;;

45)
sync_data_to_scratch
echo quantile regression, zero out RH profile for building etc.;
python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.out_channels 303 \
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
python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.zero_out_nonveg True \
        --model.init_args.backbone.init_args.out_channels 303 \
        --model.init_args.backbone.init_args.in_channels 15 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.feed_latlon True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_zero_out_geo
        ;;


47)
sync_data_to_scratch
echo quantile regression with latlon as input;
python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.zero_out_nonveg False \
        --model.init_args.backbone.init_args.out_channels 303 \
        --model.init_args.loss_fc.class_path models.losses.quantile_loss.QuantileLoss \
        --model.init_args.backbone.init_args.in_channels 15 \
        --model.init_args.feed_latlon True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_geo
        ;;
48)
sync_data_to_scratch
echo quantile regression and land cover mapping;
python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.zero_out_nonveg True \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_zero_out_lc
        ;;
49)
sync_data_to_scratch
echo quantile regression and land cover mapping;
python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 512 \
        --model.init_args.zero_out_nonveg False \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_lc
        ;;
410)
sync_data_to_scratch
echo use latlon as an input
python run.py $subcommand -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.filter_out_nonveg False \
        --model.init_args.backbone.init_args.in_channels 15 \
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
python run.py fit -c config/train.yaml --model.backbone config/model/standard_unet.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.backbone.init_args.encoder.init_args.in_channels 15 \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.loss_fc.init_args.radius 1 \
        --model.init_args.zero_out_nonveg True \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_and_LCC_zero_out_latlon_shiftloss
        ;;
31)
sync_data_to_scratch
echo using weighted loss;
python run.py fit -c config/train.yaml --model.backbone config/model/standard_unet.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --model.init_args.feed_latlon True \
        --model.init_args.backbone.init_args.encoder.init_args.in_channels 15 \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.backbone.init_args.encoder.init_args.scale_factor 2 \
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
python run.py fit -c config/train.yaml --model.backbone config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.restrict_rf False \
        --model.init_args.backbone.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.backbone.init_args.num_sepconv_blocks 8 \
        --model.init_args.backbone.init_args.activation_layer torch.nn.ReLU \
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
python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.backbone.init_args.num_sepconv_blocks 3 \
        --model.init_args.backbone.init_args.nonlinear_order $nonlinear_order \
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
python run.py fit -c config/train.yaml --model.backbone config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.backbone.init_args.num_nonlin_blocks $num_nonlin_blocks \
        --model.init_args.backbone.init_args.activation_layer torch.nn.ReLU \
        --model.init_args.backbone.init_args.mid_block models.modules.xception_blocks.SepConvBlockSkip \
        --model.init_args.backbone.init_args.mlp_skip True \
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
python run.py fit -c config/train.yaml --model.backbone config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.backbone.init_args.num_nonlin_blocks $num_nonlin_blocks \
        --model.init_args.backbone.init_args.activation_layer torch.nn.ReLU \
        --model.init_args.backbone.init_args.mid_block models.modules.xception_blocks.SepConvBlockSkip \
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
python run.py fit -c config/train.yaml --model.backbone config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.backbone.init_args.num_nonlin_blocks $num_nonlin_blocks \
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
python run.py fit -c config/train.yaml --model.backbone config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.backbone.init_args.num_nonlin_blocks $num_nonlin_blocks \
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
python run.py fit -c config/train.yaml --model.backbone config/model/xception_s2.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --model.init_args.backbone.init_args.num_sepconv_filters $num_sepconv_filters \
        --model.init_args.backbone.init_args.num_nonlin_blocks $num_nonlin_blocks \
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

    python run.py fit -c config/train.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.batch_size 1024 \
        --model.init_args.feed_latlon True \
        --model.init_args.zero_out_nonveg True \
        --model.init_args.backbone.init_args.encoder.init_args.scale_factor $scale_factor \
        --model.init_args.backbone.init_args.encoder.init_args.in_channels 15 \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --optimizer.init_args.lr $lr \
        --lr_scheduler.init_args.num_warmup_steps 6980 \
        --trainer.num_nodes 2 \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name test_warmup_steps_10_epochs
;;

# ====================================================================================
# Fast dev run
# ====================================================================================
01)
train_data_name=debug0_filtered_v1
val_data_name=debug1_filtered_v1
distributed=False
# train_data_name=train*_filtered_v1
# val_data_name=val_filtered_v1_1m
num_nonlin_blocks=3
num_sepconv_filters=256
max_epochs=1
sync_data_to_scratch
echo "$data_dir/$val_data_name.beton"
ls "$data_dir/$val_data_name.beton"
echo Using Xception;
module load gcc/11.2.0
python run.py $subcommand -c config/qat.yaml --model.backbone config/model/xception_mix_order.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --data.init_args.cal_fp "$data_dir/val_filtered_v1_1m.beton" \
        --data.init_args.distributed $distributed \
        --model.init_args.feed_latlon True \
        --model.init_args.filter_out_nonveg True \
        --model.init_args.backbone.init_args.in_channels 15 \
        --model.init_args.backbone.init_args.out_channels 315 \
        --model.init_args.loss_fc.class_path models.losses.quantile_ce_loss.QuantileCELoss \
        --quantize_model True \
        --correct_bias False \
        --trainer.max_epochs $max_epochs \
        --trainer.logger.init_args.id $run_id \
        --trainer.logger.init_args.name QR_veg_geo_lc_QAT_test
        ;;
*)
echo runnning nothing ;;
esac

echo finished
