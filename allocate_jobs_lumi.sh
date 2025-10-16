#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=allocate
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

# =======================================
#    JOB ARRAY FOR LINES, CONTROL THE MAX NUMBER OF 
# =======================================
year=2024
config_dir="${HOME}/data/GVS/Deploy/slurm_job_files_${year}"
# FILE_START=0
# FILE_END=46
FILE_LIST=($(seq 1 88))
# FILE_LIST=(59 60 61 65 69 70 85 87)
# FILE_LIST_1=($(seq 21 46))
# FILE_LIST_2=(59 60 61 65 69 70 85 87)
# FILE_LIST=("${FILE_LIST_1[@]}" "${FILE_LIST_2[@]}")
use_flash=True

# Target active inference jobs (pending + running)
TARGET_ACTIVE=180
# How many inference tasks to submit per top-up
CHUNK_SIZE=10
# Re-check interval when at capacity (seconds)
RECHECK_INTERVAL=60

check_unfinished_tiles() {
  local tile_list=("$@")
  local result=""
  for tile in "${tile_list[@]}"; do
    # sync_flag="${HOME}/data/GVS/Deploy/flags_sync_${year}/${tile}_best_images_done"
    # translate_flag="${HOME}/data/GVS/Deploy/flags_translate_${year}/${tile}_best_images_done"
    # processed_flag="${HOME}/data/GVS/Deploy/flags_sync_${year}/${tile}_done"
    inference_flag="${HOME}/data/GVS/Deploy/flags_inference_${year}/${tile}_best_images_done"
    # if [ ! -f "$sync_flag" ] && [ ! -f "$translate_flag" ] && [ ! -f "$processed_flag" ]; then
    if [ ! -f "$inference_flag" ]; then
      result+="$tile "
    fi
  done
  echo "$result"
}

# Count active (PD+R) tasks for our inference/translate jobs, expanding array ranges
count_jobs() {
  local job_name="$1"
  if [[ -z "$job_name" ]]; then
    echo "Usage: count_jobs <job_name>"
    return 1
  fi
  squeue -h --array -u "$USER" -t RUNNING,PENDING -n "$job_name" | wc -l
}

# for file_num in $(seq $FILE_START $FILE_END); do
for file_num in "${FILE_LIST[@]}"; do
  echo "Launching jobs for file $file_num, use_flash=$use_flash"
  
  tile_id_file="${config_dir}/deploy_s2_items_${year}_part${file_num}.txt"
  mapfile -t tile_array < <(cut -d, -f1 "$tile_id_file" | tail -n +2)  # read only the first item of each line into array, skipping header
  unfinished_tiles=$(check_unfinished_tiles "${tile_array[@]}")
  
  echo "Unfinished tiles: $unfinished_tiles"
  
  if [[ -z "$unfinished_tiles" ]]; then
    echo "All tiles are already translated, skipping job submission for file $file_num"
    continue
  fi

  # Build a file with only unfinished tiles (order preserved)
  tile_list_file="${config_dir}/unfinished_tiles_${year}_part${file_num}.txt"
  printf "%s\n" $unfinished_tiles > "$tile_list_file"
  
  num_tiles=$(wc -l < "$tile_list_file")
  echo "Need to submit $num_tiles tasks for file $file_num"

  # Submit in chunks while keeping ~TARGET_ACTIVE active tasks
  start_index=2 # start from line 2 because line 1 is the header
  end_index=0
  while [ $end_index -lt $num_tiles ]; do
    # Wait until we have room to submit more
    while true; do
      read active_inf_now < <(count_jobs inference)
      # read active_inf_now active_trans_now < <(count_active_tasks)
      echo "********** Active tasks: Inference=$active_inf_now"
      if [ "$active_inf_now" -lt "$TARGET_ACTIVE" ]; then
        # if [ "$active_inf_now" -lt "$TARGET_ACTIVE" ] && [ "$active_trans_now" -lt "$TARGET_ACTIVE" ]; then
        break
      fi
      echo "Active tasks ($active_inf_now) >= target ($TARGET_ACTIVE). Sleeping ${RECHECK_INTERVAL}s..."
      # echo "Active tasks ($active_inf_now, $active_trans_now) >= target ($TARGET_ACTIVE). Sleeping ${RECHECK_INTERVAL}s..."
      sleep "$RECHECK_INTERVAL"
    done

    # Determine how many tasks to submit in this top-up
    remaining=$(( num_tiles - end_index ))
    max_active=$(( active_inf_now ))
    # max_active=$(( active_inf_now > active_trans_now ? active_inf_now : active_trans_now ))
    capacity=$(( TARGET_ACTIVE - max_active ))
    submit_now=$CHUNK_SIZE
    if [ $submit_now -gt $remaining ]; then submit_now=$remaining; fi
    if [ $submit_now -gt $capacity ]; then submit_now=$capacity; fi
    if [ $submit_now -le 0 ]; then
      # Nothing to submit at the moment; re-check shortly
      sleep "$RECHECK_INTERVAL"
      continue
    fi

    start_index=$(( end_index + 1 ))
    end_index=$(( end_index + submit_now ))

    echo "Submitting inference array range ${start_index}-${end_index} (capacity=${capacity}, remaining=${remaining})"
    # JOBID_A=$(sbatch --account project_465001846 -p small --array=${start_index}-${end_index} --job-name=inference --wrap='echo "hello from $SLURM_ARRAY_TASK_ID"; sleep ${infer_time}' | awk '{print $4}')
    JOBID_A=$(sbatch --parsable --array=${start_index}-${end_index} run_lumi_inference.sh "$tile_list_file" "$file_num" "$year" "$use_flash" | awk '{print $1}')
    echo "Submitted inference job range ${start_index}-${end_index} as job $JOBID_A"
    # # sbatch --account project_465001846 -p small --array=${start_index}-${end_index} --dependency=aftercorr:${JOBID_A} --job-name=trs --wrap='echo "hello from $SLURM_ARRAY_TASK_ID"; sleep 10'
    # JOBID_B=$(sbatch --array=${start_index}-${end_index} --dependency=aftercorr:${JOBID_A} run_lumi_translate.sh "$tile_list_file" "$year" "$use_flash" | awk '{print $4}')
    # echo "Submitted translate job range ${start_index}-${end_index} after $JOBID_A as job $JOBID_B"
    # echo '--------------------------------------------------------------------'
    # Small delay to avoid overwhelming scheduler
    sleep 1s
  done

done


# # =======================================
# #    JOB ARRAY FOR CONFIG FILES
# # =======================================
# LINE_START=1
# LINE_END=72

# FILE_START=20
# FILE_END=29
# use_flash=True
# for line_num in $(seq $LINE_START $LINE_END); do
#   echo "Launching jobs for line $line_num, use_flash=$use_flash"
#   JOBID_A=$(sbatch --array=$FILE_START-$FILE_END%1 run_lumi_inference.sh $line_num 2024 $use_flash | awk '{print $4}')
#   echo "Submitted inference job for line $line_num as job $JOBID_A"
#   sbatch --array=$FILE_START-$FILE_END%1 --dependency=aftercorr:${JOBID_A} run_lumi_translate.sh $line_num 2024 $use_flash
#   echo "Submitted translate job for line $line_num"
# done


# use_flash=False
# FILE_START=30
# FILE_END=39
# for line_num in $(seq $LINE_START $LINE_END); do
#   echo "Launching jobs for line $line_num, use_flash=$use_flash"
#   JOBID_A=$(sbatch --array=$FILE_START-$FILE_END%1 run_lumi_stream.sh $line_num 2024 $use_flash | awk '{print $4}')
#   echo "Submitted stream job for line $line_num as job $JOBID_A"
#   JOBID_B=$(sbatch --array=$FILE_START-$FILE_END%1 --dependency=aftercorr:${JOBID_A} run_lumi_inference.sh $line_num 2024 $use_flash | awk '{print $4}')
#   echo "Submitted inference job for line $line_num as job $JOBID_B"
#   sbatch --array=$FILE_START-$FILE_END%1 --dependency=aftercorr:${JOBID_B} run_lumi_translate.sh $line_num 2024 $use_flash
#   echo "Submitted translate job for line $line_num"
# done



# =======================================
#    PREDICT MULTIPLE TILES IN ONE JOB
# =======================================
# for i in $(seq $START $END); do
#   echo "Launching jobs for line $i, use_flash=$use_flash"
#
#   # Submit data streaming job (CPU)
#   JOB1_ID=$(sbatch --parsable \
#     --export=TASK_ID=$i \
#     --job-name=stream${i} \
#     run_lumi_stream.sh $i 2024 $use_flash)
#
#   echo "Submitted job1 line $i as job $JOB1_ID"
#
#   # Submit inference (GPU) after data streaming starts
#   JOB2_ID=$(sbatch --parsable \
#     --dependency=after:$JOB1_ID \
#     --export=TASK_ID=$i \
#     --job-name=inf${i} \
#     run_lumi_inference.sh $i 2024 $use_flash)
#
#   echo "Submitted job2 line $i as job $JOB2_ID"
#
#   # Submit translate (CPU) after inference starts
#   sbatch \
#     --dependency=after:$JOB2_ID \
#     --export=TASK_ID=$i \
#     --job-name=trans${i} \
#     run_lumi_translate.sh $i 2024 $use_flash
#
#   echo "Submitted job3 line $i"
# done