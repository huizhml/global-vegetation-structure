
a100_nodes=(hendrixgpu01fl hendrixgpu02fl)
l40s_nodes=(hendrixgpu23fl hendrixgpu24fl hendrixgpu25fl hendrixgpu26fl)
year=2024
config_dir="${HOME}/data/GVS/deploy/slurm_job_files_${year}"

FILE_LIST=$(seq 0 22)
TARGET_ACTIVE=24
# How many inference tasks to submit per top-up
CHUNK_SIZE=4
# Re-check interval when at capacity (seconds)
RECHECK_INTERVAL=60

check_unfinished_tiles() {
  local tile_list=("$@")
  local result=""
  for tile in "${tile_list[@]}"; do
    translate_flag="${HOME}/data/GVS/deploy/translate_flags_${year}/${tile}_done"
    translate_flag_new="${HOME}/data/GVS/deploy/translate_flags_${year}/${tile}_best_images_done"
    if [ ! -f "$translate_flag" ] && [ ! -f "$translate_flag_new" ]; then
      result+="$tile "
    fi
  done
  echo "$result"
}

job_array_id=$(sbatch --array=2-13 run_deploy.sh 2020 | awk '{print $4}')
sbatch --array=2-13 --dependency=aftercorr:${job_array_id} --cpus-per-task=8 run_cpu.sh 2020

job_array_id=$(sbatch --array=2-25 run_deploy.sh 2024 | awk '{print $4}')
sbatch --array=2-25 --dependency=aftercorr:${job_array_id} --cpus-per-task=8 run_cpu.sh 2024


# # launch jobs for hendrixgpu01fl
# for idx in $(seq 0 6); do
#   echo "Launching jobs for index $idx on hendrixgpu01fl"
#   sbatch -w hendrixgpu01fl run_deploy.sh $idx 2020
#   sbatch -w hendrixgpu01fl --cpus-per-task=8 run_cpu.sh $idx 2020
# done

# # launch jobs for hendrixgpu02fl
# for idx in $(seq 7 8); do
#   echo "Launching jobs for index $idx on hendrixgpu02fl"
#   sbatch -w hendrixgpu02fl run_deploy.sh $idx 2020
#   sbatch -w hendrixgpu02fl --cpus-per-task=16 run_cpu.sh $idx 2020
# done

# # launch jobs for l40s
# part_idx=8
# for node in ${l40s_nodes[@]}; do
#   for idx in $(seq 0 3); do
#     part_idx=$((part_idx + 1))
#     echo "Launching jobs for index $part_idx on $node"
#     sbatch -w $node run_deploy.sh $part_idx 2020  
#     sbatch -w $node --cpus-per-task=4 run_cpu.sh $part_idx 2020
#   done
# done

# # launch jobs for hendrixgpu26fl
# for idx in $(seq 0 3); do
#     echo "Launching jobs for index $idx on hendrixgpu26fl"
#     sbatch -w hendrixgpu26fl run_deploy.sh $idx 2024
#     sbatch -w hendrixgpu26fl --cpus-per-task=4 run_cpu.sh $idx 2024
# done



# ================== 2024 ==================

# #launch jobs for hendrixgpu01fl
# for idx in $(seq 66 72); do
#   echo "Launching jobs for index $idx on hendrixgpu01fl"
#   sbatch -w hendrixgpu01fl run_deploy.sh $idx 2024
#   sbatch -w hendrixgpu01fl --cpus-per-task=8 run_cpu.sh $idx 2024
# done

# # launch jobs for hendrixgpu02fl
# for idx in $(seq 73 74); do
#   echo "Launching jobs for index $idx on hendrixgpu02fl"
#   sbatch -w hendrixgpu02fl run_deploy.sh $idx 2024
#   sbatch -w hendrixgpu02fl --cpus-per-task=16 run_cpu.sh $idx 2024
# done

# # launch jobs for l40s
# part_idx=74
# for node in ${l40s_nodes[@]}; do
#   for idx in $(seq 0 3); do
#     part_idx=$((part_idx + 1))
#     echo "Launching jobs for index $part_idx on $node"
#     sbatch -w $node run_deploy.sh $part_idx 2024  
#     sbatch -w $node --cpus-per-task=4 run_cpu.sh $part_idx 2024
#   done
# done

# # launch jobs for hendrixgpu26fl
# for idx in $(seq 47 49); do
#     echo "Launching jobs for index $idx on hendrixgpu01fl"
#     # sbatch -w hendrixgpu01fl run_deploy.sh $idx 2024
#     sbatch -w hendrixgpu01fl --cpus-per-task=8 run_cpu.sh $idx 2024
# done

# for idx in $(seq 66 68); do
#     echo "Launching jobs for index $idx on hendrixgpu01fl"
#     # sbatch -w hendrixgpu01fl run_deploy.sh $idx 2024
#     sbatch -w hendrixgpu01fl --cpus-per-task=8 run_cpu.sh $idx 2024
# done