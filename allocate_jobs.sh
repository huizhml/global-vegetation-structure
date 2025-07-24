
a100_nodes=(hendrixgpu01fl hendrixgpu02fl)
l40s_nodes=(hendrixgpu23fl hendrixgpu24fl hendrixgpu25fl)

# launch jobs for hendrixgpu01fl
for idx in $(seq 0 6); do
  echo "Launching jobs for index $idx on hendrixgpu01fl"
#   sbatch -w hendrixgpu01fl run_deploy.sh $idx 2020
#   sbatch -w hendrixgpu01fl --cpus-per-task=8 run_cpu.sh $idx 2020
done

# launch jobs for hendrixgpu02fl
for idx in $(seq 7 8); do
  echo "Launching jobs for index $idx on hendrixgpu02fl"
#   sbatch -w hendrixgpu02fl run_deploy.sh $idx 2020
#   sbatch -w hendrixgpu02fl --cpus-per-task=16 run_cpu.sh $idx 2020
done

# launch jobs for l40s
part_idx=8
for node in ${l40s_nodes[@]}; do
  for idx in $(seq 0 3); do
    part_idx=$((part_idx + 1))
    echo "Launching jobs for index $part_idx on $node"
    # sbatch -w $node run_deploy.sh $part_idx 2020  
    # sbatch -w $node --cpus-per-task=4 run_cpu.sh $part_idx 2020
  done
done

# launch jobs for hendrixgpu26fl
for idx in $(seq 0 3); do
    echo "Launching jobs for index $idx on hendrixgpu26fl"
    # sbatch -w hendrixgpu26fl run_deploy.sh $idx 2024
    # sbatch -w hendrixgpu26fl --cpus-per-task=4 run_cpu.sh $idx 2024
done



