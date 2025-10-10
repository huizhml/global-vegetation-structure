# Prepare data

## Train-val-cal-test split
1. Generate index table for all the data downloaded
    This will walk through all {zone}.h5 files and generate the index table
    [workflow](obsidian://open?vault=MyIOTO&file=4-%E6%88%90%E6%9E%9C%2FGlobalVegetationStructure%2FSummary%2Ftrain-val-cal-test%20split.excalidraw)
    ```bash
    python -m datasets.gen_index_table  h5_dir=~/data/GEDI
    ```
    **Output**: ~/data/geo_index_table/{zone}.parquet [lumi]
2. Split the index table into train/val/cal/test

    ```bash
    python -m datasets._data_split
    ```
3. Split h5 files into train/val/cal/test

## Split train data into $k$ subsets
1. Reindex the `in_partition_idx in` train index table 
2. Random split train index table
3. 


```python
python -m datasets._split_train out_dir=/users/zhanghui/flash/data/ h5_file=/users/zhanghui/flash/data/train.h5
```

# Spatial uniform sampling with high sensitivity GEDI data
## Download GEDI points with sensitivity >= 0.95
### First pass
sbatch download/download_gedi.sh 2019 "keys/private-key.json"  7561  15h
sbatch download/download_gedi.sh 2020 "keys/nrt-key.json"      7568  22h
sbatch download/download_gedi.sh 2021 "keys/feng-key.json"     7569  20h
sbatch download/download_gedi.sh 2022 "keys/venky-key.json"    7574  21h

### Second pass
NOTE: some failed because of the request rate limit
sbatch download/download_gedi.sh 2019 "keys/private-key.json"  bash
sbatch download/download_gedi.sh 2020 "keys/nrt-key.json"      7918 35min
sbatch download/download_gedi.sh 2021 "keys/feng-key.json"     7919 36min
sbatch download/download_gedi.sh 2022 "keys/venky-key.json"    7920 1h

## Figure out which GEDI points we need to download additionaly
NOTE: this step could be merged with the last step. It's seperated because of bugs
sbatch download/download_gedi.sh 2019 
sbatch download/download_gedi.sh 2020 7937  1h
sbatch download/download_gedi.sh 2021 7938  1h
sbatch download/download_gedi.sh 2022 7939  1h

## Gather S2 meta
sbatch --array=2-17 download/download_s2.sh 2 
NOTE: using config file correct_job_config.txt
- First Pass: failed, took 30+h
  Exception: "FileNotFoundError('An error occurred while calling the read_parquet method registered to the pandas backend.\\nOriginal Message: items/sentinel-2-l2a.parquet/part-0331_2021-10-25T10:25:31+00:00_2021-11-01T10:25:31+00:00.parquet')"

## Find best S2
sbatch --mem 256G -p small --array=18-18 download/download_s2.sh 3
sbatch --array=18-18 download/download_s2.sh 3

## Download

python -m download.s2_download zone=52M \
           gedi_dir=${HOME}/data/GVS/GEDI_extra_with_s2_candidates_and_best \
           save_dir=${HOME}/data/GVS/GEDI_S2_h5_extra \
           flag_dir=${HOME}/data/GVS/extra_download_flags
  

# Train

## Debug training
1. Turn model logging off, otherwise get error 'too many opened files'
2. log less often to wandb, eg., trainer.log_every_n_steps=1000, otherwise get error 'too many opened files'
exp.
```bash
train_data_name=debug0_filtered_v0
val_data_name=val_filtered_v0_1m
data_dir=${HOME}/data/GVS/train_subsets
python run.py fit -c config/train.yaml \
        --data.init_args.train_fp "$data_dir/$train_data_name.beton" \
        --data.init_args.val_fp "$data_dir/$val_data_name.beton" \
        --trainer.logger.init_args.name L1_loss \
        --trainer.log_every_n_steps 100 \
        --trainer.max_epochs 200 \
        --trainer.limit_train_batches 10 \
        --trainer.limit_val_batches 10
```


# Scripts
1. Check how long a completed job took
    sacct -j <job_id> --format=JobID,JobName,Elapsed,State
2. Check previous jobs
    sacct --starttime=now-7days --user=$USER --format=JobID,JobName,State,Elapsed,ExitCode
3. Sync data between clusters
   ```bash
    TARGET_HOST="zhanghui@lumi.csc.fi"
    TARGET_DIR="/users/zhanghui/data/GVS/GEDI_extra_with_s2_candidates"
    SOURCE_DIR="/home/ksb781/data/GVS/GEDI_extra_with_s2_candidates"
    rsync -avz --progress "$SOURCE_DIR/2019" "$TARGET_HOST:$TARGET_DIR"
    ```

# LUMI ENV
[Tutorial](https://lumi-supercomputer.github.io/LUMI-training-materials/2day-20240502/Demo1/#step-4-adding-python-packages)

module load LUMI/24.03 partition/container EasyBuild-user
1. copy the config file 
    ```bash
eb --copy-ec PyTorch-2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617.eb PyTorch-2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617-ffcv.eb
```
2. change suffix

```bash
    sed -e "s|^\(versionsuffix.*\)-singularity-20240617|\1-singularity-20240617-ffcv|" -i PyTorch-2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617-ffcv.eb
```

check
```bash
grep versionsuffix PyTorch-2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617-ffcv.eb
```

cat > lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2-ffcv.def <<EOF

Bootstrap: localimage

From: $CONTAINERFILE

%post

zypper -n install -y Mesa libglvnd libgthread-2_0-0 hostname

EOF

cat lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2-ffcv.def


GLib/2.78.1-cpeCray-24.03