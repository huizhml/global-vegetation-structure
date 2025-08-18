#!/bin/bash
##SBATCH --account=project_465000894
#SBATCH --partition=ml4good
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=1-00:00:00
#SBATCH --job-name=submit
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu01fl,hendrixgpu02fl,hendrixgpu03fl,hendrixgpu04fl,hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu08fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl,hendrixgpu21fl
hostname
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Number of tasks: $SLURM_NTASKS"
echo "Time requested: $SLURM_TIMELIMIT"
scontrol show job $SLURM_JOB_ID | grep "TRES="

data_dir=${HOME}/data/GEDI

id=$1
echo running job $id;
case $id in
1)
echo generate index table for the whole downloaded data
python -m datasets.1_generate_index_table;;
2)
echo split index table and h5 files into train, val, test, cal
#! incountinous write won't work, there will be some h5 files corrupted
echo We split the {zone}.h5 files here, becase converting h5 file to {train/val/cal/test}.beton based on the large h5 is inefficient.
python -m datasets._2_train_test_split index_dir=$data_dir/geo_index_table_with_sensitivity;;
3)
echo merge {split} h5 files, the input for converting to beton
for split in train val cal test; do
        python -m datasets._3_merge_h5s h5_dir="$data_dir/split_test0.1_cal0.1_val0.1_seed42_v1/${split}_h5s" \
                merged_h5_file="$data_dir/${split}.h5"
done;;
4)
# USE node 22, took ~30min to convert one subset, needs 240GB memory
nsplit=10
debug=${2:-False}
idx=$SLURM_ARRAY_TASK_ID
echo split training data index into $nsplit subsets, and converting to beton using FFCV
input_dir=$data_dir
# if hostname | grep -q "hendrix"; then
#         echo "Running on Hendrix"
#         if [ ! -f /scratch/train.h5 ]; then
#                 rsync -av --progress $data_dir/train.h5 /scratch/
#         else
#                 echo "train.h5 already exists in /scratch"
#         fi
#         input_dir=/scratch
# else
#         echo "Running on LUMI"
#         input_dir=$data_dir
# fi

hostname=$(hostname)
number=$(echo "$hostname" | grep -o '[0-9]\+')
echo "Number is $number"
if hostname | grep -q "hendrix"; then
        host_list=("01" "02" "07" "16" "22" "23" "24" "25" "26")
        if [[ " ${host_list[@]} " =~ " $number " ]]; then
                echo "node $number has scratch folder"
                input_dir=/scratch
                rsync -av --progress ${HOME}/data/GVS/cal.h5 $input_dir/cal.h5
                
        else
                echo "node $number does not have scratch folder"
                input_dir=${HOME}/data/GVS
        fi
        fi

# python -m datasets._4_convert_to_beton nsplit=$nsplit split_idx=$idx \
#         h5_file=$input_dir/train.h5 out_idx_dir=$data_dir/index_table_train_subsets \
#         index_table=$data_dir/split_test0.1_cal0.1_val0.1_seed42_v1/index_table_train \
#         shuffle_indices=True +debug=$debug version=$3

python -m datasets._4_convert_to_beton \
        h5_file=$input_dir/cal.h5 \
        index_table=$data_dir/split_test0.1_cal0.1_val0.1_seed42_v1/index_table_cal \
        shuffle_indices=False +debug=$debug version=1 +create_subset=False
;;

5)
echo calculate the mean and std of sentinel-2 images from the training data
python -m datasets._5_calculate_stats;;

6)
echo get the mean and std from the training data
python -m datasets.ffcv_datamodule data.init_args.train_fp="~/data/GEDI/train.beton" \
        data.init_args.order=SEQUENTIAL data.init_args.batch_size=2048 \
        data.init_args.num_workers=32 ;;

7)
echo random sample one val subset, 1/10 of the original val set;
#echo merge all h5 files;
#python -m datasets._merge_h5s h5_dir="$data_dir/GEDI_S2_h5s_original" merged_h5_file="$data_dir/GVS.h5" functions=['merge_all_zones']
echo generate val subset index;

python -m datasets._convert_to_beton \
        split_dir="$data_dir/split_test0.1_cal0.1_val0.1_seed42" \
        h5_file="$data_dir/GVS.h5" out_dir="$data_dir/train_subsets" splits=['val'] \
        +subset=True ;;
8)
echo Run PCA on the training subset;
python -m datasets.statistical_analysis \
        data_fps=$data_dir'/train_subsets/train*_attrs.beton' \
        model_path=output/pca_model.pkl;;
9)
echo aggregate the GEDI data;
python -m datasets._6_visual_check task=aggregate_gedi_by_biome beton_fps=${data_dir}/train_subsets/test*_v3.beton
;;
10)
echo download inference data by api query;
python -m download._5_download_inference task=download_by_api_query year=2024 specified_tiles_file=~/data/GVS/Deploy/tiles_without_images_and_metadata_2024.txt;;
11)
echo download inference data by metadata;
python -m download._5_download_inference task=download year=2024 job_id=0;;
*)
echo runnning nothing ;;
esac

echo finished job $id
