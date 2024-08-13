#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small
#SBATCH --cpus-per-task=32
#SBATCH --mem=320GB
#SBATCH --time=3-00:00:00
#SBATCH --job-name=submit
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu08fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch
hostname
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Number of tasks: $SLURM_NTASKS"
echo "Time requested: $SLURM_TIMELIMIT"
scontrol show job $SLURM_JOB_ID | grep "TRES="
# python -m datasets._split_train
id=$1
case $id in
1)
echo running job 1 ;
echo merge train h5 files
python -m datasets._merge_h5s merged_h5_file='~/flash/data/train.h5';;

2)
echo running job 2 ;
nsplit=10
debug=${2:-False}
idx=$SLURM_ARRAY_TASK_ID
echo split training data into $nsplit subsets, using FFCV
python -m datasets._split_train nsplit=$nsplit split_idx=$idx shuffle_indices=True +debug=$debug;;

3)
echo aggregate RHs boxplot stats and visualize
python -m datasets.stats task=plot_boxplots splits=['train','val'] +boxplot_dir="~/scratch/data/boxplots";;

4)
echo get the mean and std from the training data
python -m datasets.ffcv_datamodule data.init_args.train_fp="~/flash/data/val.beton" \
        data.init_args.order=SEQUENTIAL data.init_args.batch_size=2048 \
        data.init_args.num_workers=32 ;;

*)
echo runnning nothing ;;
esac

echo finished job $id