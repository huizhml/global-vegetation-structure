START=0
END=0

for i in $(seq $START $END); do
  echo "Launching jobs for index $i"

  # Submit data streaming job (CPU)
  JOB1_ID=$(sbatch --parsable \
    --export=TASK_ID=$i \
    run_lumi_stream.sh $i 2024)

  echo "Submitted job1 index $i as job $JOB1_ID"

  # Submit inference (GPU) after data streaming starts
  JOB2_ID=$(sbatch --parsable \
    --dependency=after:$JOB1_ID \
    --export=TASK_ID=$i \
    run_lumi_inference.sh $i 2024)

  echo "Submitted job2 index $i as job $JOB2_ID"

  # Submit translate (CPU) after inference starts
  sbatch \
    --dependency=after:$JOB2_ID \
    --export=TASK_ID=$i \
    run_lumi_translate.sh $i 2024

  echo "Submitted job3 index $i"
done