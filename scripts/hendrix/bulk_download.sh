list_file=${HOME}/gvsm/assets/worklists/dk.txt
save_dir=/Volumes/Seagate/vsm/


case $1 in
1)
# ---------------------------------------
#   Download global single layer
# ---------------------------------------
# This organizes the data by RH98_Q{q_idx}/{tile_id}.tif
for q_idx in 1 2 0; do
    dst_dir=${save_dir}/RH98_Q${q_idx}
    mkdir -p $dst_dir
    total=$(wc -l < "$list_file" | tr -d ' ')
    start_time=$(date +%s)
    count=0
    while read line; do
        case "$line" in DONE:*) ;; *) continue ;; esac
        tile="${line#DONE:}"
        count=$((count + 1))
        now=$(date +%s)
        elapsed=$((now - start_time))
        spt=$(echo "scale=1; $elapsed / $count" | bc)
        remaining=$(echo "scale=0; ($total - $count) * $spt / 1" | bc)
        mins=$((remaining / 60))
        secs=$((remaining % 60))
        echo "[$count/$total] $tile | ${spt}s/tile | ETA: ${mins}m${secs}s"
    done < <(cat $list_file | xargs -P 8 -I {} bash -c \
        'rsync -az "hendrix1:/home/ksb781/data/gvs/predictions/2020/blended/tiles/cog/{}/RH98_Q${2}.tif" "$1/{}.tif" 2>/dev/null && echo "DONE:{}"' \
        _ "$dst_dir" "$q_idx")
done
    ;;
2)
# ---------------------------------------
#   Download local full VSM, 1 SSH session, 8 rsync processes
# ---------------------------------------
# This doesn't change the directory structure
dst_dir=${save_dir}/2020
mkdir -p ${dst_dir}/

while read line; do
    case "$line" in DONE:*) ;; *) continue ;; esac
    tile="${line#DONE:}"
    mkdir -p ${dst_dir}/${tile}
    count=$((count + 1))
    now=$(date +%s)
    elapsed=$((now - start_time))
    spt=$(echo "scale=1; $elapsed / $count" | bc)
    remaining=$(echo "scale=0; ($total - $count) * $spt / 1" | bc)
    mins=$((remaining / 60))
    secs=$((remaining % 60))
    echo "[$count/$total] $tile | ${spt}s/tile | ETA: ${mins}m${secs}s"
done < <(cat $list_file | xargs -P 8 -I {} bash -c \
    'rsync -az "hendrix1:/home/ksb781/data/gvs/predictions/2020/blended/tiles/cog/{}/*.tif" "$1/{}/" 2>/dev/null && echo "DONE:{}"' \
    _ "$dst_dir")
    ;;

3)
# ---------------------------------------
#   Download local full VSM, 4 parallel SSH sessions
# ---------------------------------------
# This doesn't change the directory structure
dst_dir=${save_dir}/2020
mkdir -p ${dst_dir}/

read -s -p "Enter hendrix1 password: " SSHPASS
echo
export SSHPASS

count=0
start_time=$(date +%s)
total=$(wc -l < "$list_file")
chunk_size=$(( (total + 3) / 4 ))
split -l "$chunk_size" "$list_file" /tmp/chunk_
chunks=(/tmp/chunk_*)

(
    for i in 0 1 2 3; do
        chunk="${chunks[$i]}"
        socket="/tmp/ssh_socket_${i}"

        # Open persistent connection using password
        sshpass -e ssh -MNf \
            -o ControlPath="$socket" \
            -o ControlPersist=yes \
            -o StrictHostKeyChecking=no \
            ksb781@hendrix1

        (
            while read tile; do
                rsync -a --no-compress \
                    -e "ssh -o ControlPath=${socket} -o StrictHostKeyChecking=no" \
                    "hendrix1:/home/ksb781/data/gvs/predictions/2020/original/tiles/cog/${tile}/" \
                    "${dst_dir}/${tile}/" 2>/dev/null \
                && echo "DONE:${tile}"
            done < "$chunk"

            ssh -O exit -o ControlPath="$socket" hendrix1 2>/dev/null
        ) &
    done
    wait
) | while read line; do
    case "$line" in DONE:*) ;; *) continue ;; esac
    tile="${line#DONE:}"
    count=$((count + 1))
    now=$(date +%s)
    elapsed=$((now - start_time))
    [ "$elapsed" -eq 0 ] && elapsed=1
    spt=$(echo "scale=1; $elapsed / $count" | bc)
    remaining=$(echo "scale=0; ($total - $count) * $spt / 1" | bc)
    mins=$((remaining / 60))
    secs=$((remaining % 60))
    echo "[$count/$total] $tile | ${spt}s/tile | ETA: ${mins}m${secs}s"
done

unset SSHPASS
;;

4)
# ---------------------------------------
#   Download local full VSM, using rclone, 2024
# ---------------------------------------
# This doesn't change the directory structure
LUMI_PROJECT=465001846
base_remote="lumi-${LUMI_PROJECT}-private:"
dst_dir="${save_dir}/2024"
year=2024

export count=0
export start_time=$(date +%s)
export total=$(wc -l < "$list_file")

echo "Downloading full VSM from LUMI-O to ${dst_dir}"
echo "Total tiles: $total"

while read line; do
    case "$line" in DONE:*) ;; *) continue ;; esac
    tile="${line#DONE:}"
    count=$((count + 1))
    now=$(date +%s)
    elapsed=$((now - start_time))
    [ "$elapsed" -eq 0 ] && elapsed=1
    spt=$(echo "scale=1; $elapsed / $count" | bc)
    remaining=$(echo "scale=0; ($total - $count) * $spt / 1" | bc)
    mins=$(( remaining / 60 ))
    secs=$(( remaining % 60 ))
    echo "[$count/$total] $tile | ${spt}s/tile | ETA: ${mins}m${secs}s"
done < <(
    cat "$list_file" | xargs -P 8 -I {} bash -c '
        tile="$1"
        year="$2"
        base_remote="$3"
        dst_dir="$4"
        zone=$(echo "${tile:0:3}" | tr "[:upper:]" "[:lower:]")
        bucket_name="${zone}-${year}"
        remote="${base_remote}${bucket_name}/predictions_GTiff_${year}/${tile}"
        mkdir -p "${dst_dir}/${tile}"
        rclone copy "$remote" "${dst_dir}/${tile}" \
            --transfers=8 --checkers=8 --multi-thread-streams=4 \
            2>/dev/null && echo "DONE:${tile}"
    ' _ {} "$year" "$base_remote" "$dst_dir"
)

now=$(date +%s)
elapsed=$(( now - start_time ))
echo "Done! Total time: $(( elapsed / 60 ))m$(( elapsed % 60 ))s"
;;
esac
