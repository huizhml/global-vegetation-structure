# postprocess blending
sbatch -w hendrixgpu01fl --ntasks-per-node=6 run_hendrix_ntasks.sh 0 0 key_rhs
sbatch -w hendrixgpu02fl --ntasks-per-node=6 run_hendrix_ntasks.sh 0 144 key_rhs
sbatch -w hendrixgpu11fl --ntasks-per-node=3 run_hendrix_ntasks.sh 0 288 key_rhs
sbatch -w hendrixgpu12fl --ntasks-per-node=3 run_hendrix_ntasks.sh 0 360 key_rhs
sbatch -w hendrixgpu23fl --ntasks-per-node=2 run_hendrix_ntasks.sh 0 432 key_rhs
sbatch -w hendrixgpu24fl --ntasks-per-node=2 run_hendrix_ntasks.sh 0 480 key_rhs
sbatch -w hendrixgpu25fl --ntasks-per-node=2 run_hendrix_ntasks.sh 0 528 key_rhs
sbatch -w hendrixgpu26fl --ntasks-per-node=2 run_hendrix_ntasks.sh 0 576 key_rhs