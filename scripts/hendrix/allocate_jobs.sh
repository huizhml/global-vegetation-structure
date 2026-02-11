
check_unfinished_tiles() {
    local year=$1
    local offset=$2
    local n_zones=$3
    local all_zones=($(ls ${HOME}/data/gvs/assets/worklists/tiles_by_mgrs_zone/*.txt | sort))
    zones=(${all_zones[@]:offset:n_zones})
    unfinished_tiles=()
    for tile_id_file in ${zones[@]}; do
        for tile_id in $(cat $tile_id_file); do
            if [ ! -f "${HOME}/data/gvs/state/${year}/corrected/key_rhs/${tile_id}_done" ]; then
                unfinished_tiles+=($tile_id)
            fi
        done
    done
    echo ${unfinished_tiles[@]}
}

case $1 in
0)
# ==========================================
#   Initial job allocation on hendrix for postprocess
# ==========================================
sbatch -w hendrixgpu01fl --ntasks-per-node=6 run_hendrix_ntasks.sh 0 0 2020 rest_rhs
# sbatch -p gpu -t 2-00:00:00 -w hendrixgpu22fl --ntasks-per-node=6 run_hendrix_ntasks.sh 0 0 2020 rest_rhs
sbatch -w hendrixgpu02fl --ntasks-per-node=6 run_hendrix_ntasks.sh 0 144 2020 rest_rhs
sbatch -w hendrixgpu11fl --ntasks-per-node=3 run_hendrix_ntasks.sh 0 288 2020 rest_rhs
sbatch -w hendrixgpu12fl --ntasks-per-node=3 run_hendrix_ntasks.sh 0 360 2020 rest_rhs
sbatch -w hendrixgpu23fl --ntasks-per-node=2 run_hendrix_ntasks.sh 0 432 2020 rest_rhs
sbatch -w hendrixgpu24fl --ntasks-per-node=2 run_hendrix_ntasks.sh 0 480 2020 rest_rhs
sbatch -w hendrixgpu25fl --ntasks-per-node=2 run_hendrix_ntasks.sh 0 528 2020 rest_rhs
sbatch -w hendrixgpu26fl --ntasks-per-node=2 run_hendrix_ntasks.sh 0 576 2020 rest_rhs
# sbatch -w hendrixgpu01fl --ntasks-per-node=2 run_hendrix_ntasks.sh 0 576 2020 rest_rhs
;;

1)
# ==========================================
#   check unfinished tiles for postprocess
# ==========================================

offset=0
node=hendrixgpu11fl
n_zones=144
year=2020
unfinished_tiles=($(check_unfinished_tiles $year $offset $n_zones))
echo "Total unfinished tiles: ${#unfinished_tiles[@]}"
echo ${unfinished_tiles[@]}

sbatch -w $node --ntasks-per-node=3 run_hendrix_ntasks.sh 1 "${unfinished_tiles[*]}" $year
;;
2)

# ==========================================
#   check unfinished tiles for postprocess
# ==========================================

offset=144
node=hendrixgpu11fl
n_zones=144
year=2020
unfinished_tiles=($(check_unfinished_tiles $year $offset $n_zones))
echo "Total unfinished tiles: ${#unfinished_tiles[@]}"
echo ${unfinished_tiles[@]}

sbatch -w $node --ntasks-per-node=1 run_hendrix_ntasks.sh 1 "${unfinished_tiles[*]}" $year
;;

3)

offset=288
node=hendrixgpu22fl
n_zones=72
year=2020
unfinished_tiles=($(check_unfinished_tiles $year $offset $n_zones))
echo "Total unfinished tiles: ${#unfinished_tiles[@]}"
echo ${unfinished_tiles[@]}

# sbatch -p gpu -t 2-00:00:00 -w $node --ntasks-per-node=6 run_hendrix_ntasks.sh 1 "${unfinished_tiles[*]}" $year
;;

4)

# ==========================================
#   check unfinished tiles for postprocess
# ==========================================

offset=360
node=hendrixgpu11fl
n_zones=72
year=2020
unfinished_tiles=($(check_unfinished_tiles $year $offset $n_zones))
echo "Total unfinished tiles: ${#unfinished_tiles[@]}"
echo ${unfinished_tiles[@]}

# sbatch -w $node --ntasks-per-node=3 run_hendrix_ntasks.sh 1 "${unfinished_tiles[*]}" $year
;;


5)
# ==========================================
#   check unfinished tiles for postprocess
# ==========================================

offset=432
node=hendrixgpu23fl
n_zones=48
year=2020
unfinished_tiles=($(check_unfinished_tiles $year $offset $n_zones))
echo "Total unfinished tiles: ${#unfinished_tiles[@]}"
echo ${unfinished_tiles[@]}

sbatch -w $node --ntasks-per-node=1 run_hendrix_ntasks.sh 1 "${unfinished_tiles[*]}" $year

;;


6)

# ==========================================
#   check unfinished tiles for postprocess
# ==========================================

offset=480
node=hendrixgpu24fl
n_zones=48
year=2020
unfinished_tiles=($(check_unfinished_tiles $year $offset $n_zones))
echo "Total unfinished tiles: ${#unfinished_tiles[@]}"
echo ${unfinished_tiles[@]}

# sbatch -w $node --ntasks-per-node=1 run_hendrix_ntasks.sh 1 "${unfinished_tiles[*]}" $year
;;


7)

# ==========================================
#   check unfinished tiles for postprocess
# ==========================================

offset=528
node=hendrixgpu02fl
n_zones=48
year=2020
unfinished_tiles=($(check_unfinished_tiles $year $offset $n_zones))
echo "Total unfinished tiles: ${#unfinished_tiles[@]}"
echo ${unfinished_tiles[@]}

sbatch -w $node --ntasks-per-node=5 run_hendrix_ntasks.sh 1 "${unfinished_tiles[*]}" $year
;;

8)
# ==========================================
#   check unfinished tiles for postprocess
# ==========================================

offset=576
node=hendrixgpu25fl
n_zones=48
year=2020
unfinished_tiles=($(check_unfinished_tiles $year $offset $n_zones))
echo "Total unfinished tiles: ${#unfinished_tiles[@]}"
echo ${unfinished_tiles[@]}

sbatch -w $node --ntasks-per-node=1 run_hendrix_ntasks.sh 1 "${unfinished_tiles[*]}" $year
;;
9)
# ==========================================
#   get correction stats for all tiles 2020
# ==========================================
year=2020
tile_ids=($(grep '^43S' ${HOME}/data/gvs/assets/worklists/tiles_2020.txt))
for tile_id in ${tile_ids[@]}; do
    echo "Processing tile $tile_id"
    python -m postprocess.bias_correction year=$year tile_id=$tile_id task=get_correction_stats
done
;;

esac