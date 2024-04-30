#!/bin/bash

submit_job() {
  sub="$(sbatch "$@")"

  if [[ "$sub" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    exit 1
  fi
}

year=$1
flag_file="/users/zhanghui/GEDI/$year/done"
if [ ! -f "$config_file" ]; then
    echo "Continue downloading GEDI data for $year..."
    submit_job()
    exit 1
fi
# first job - no dependencies
offset=10
id1=$(submit_job --array=1-10 job1.sh)

# Two jobs that depend on the first job
id2=$(submit_job --dependency=aftercorr:$id1 --array=1-10 download/download_s2.sh 10) #11-20
id3=$(submit_job --dependency=aftercorr:$id2 --array=1-10 download/download_s2.sh 20) #21-30
# .....

# One job that depends on both the second and the third jobs
id4=$(submit_job  --dependency=afterany:$id2:$id3 job4.sh)
