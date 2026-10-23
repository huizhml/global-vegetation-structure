# VSM release manifest：released_v0

一个 release 年份实际用的是哪些文件，写成一张表。下载、分析、发布都读这张表，
不用再翻 STAC catalog，也不用记每年的目录布局。
每个版本目录里实际有什么见 [vsm_snapshot.md](vsm_snapshot.md)。

## 位置

Hendrix：`~/data/gvs/products/vsm/<year>/released_v0.tsv`，和各版本目录同级。

| 年份 | 来源 | 说明 |
|---|---|---|
| 2020 | STAC `gvsm_stac_catalog/vsm_local_masked` 的 asset href | 发到 Source Coop 的 collection。item 分散在 `original/tiles/{cog,geotiff}` 和 `masked/tiles/geotiff` 三个 root；Q1 已转 COG，Q0/Q2 还在 geotiff/ |
| 2024 | 目录 `2024/bias_corrected/tiles/cog/<TILE>/` | 不在任何 collection 里；文件名 `RH*_Q*.tif`。2026-10-23 前指向 `masked/`（另一次预测，已删）。目前只有 Q1，Q0/Q2 等 conformal 校准后才放进来，在那之前清单会报每个 tile 缺文件 |

来源定义在 `config/tools/config.yaml` 的 `release_manifest.sources`。其他年份没有 release，用 `original/`。

## 格式

不压缩的 TSV，一个 tile 一行，可以和其他按 tile 的表直接对齐：

```
tile_id  q0_root                    q1_root                q2_root                    file_suffix  n_files  size_bytes
32UNG    2020/masked/tiles/geotiff  2020/masked/tiles/cog  2020/masked/tiles/geotiff  .tif         303      ...
```

- 分位数 Q`q` 的 101 个文件是 `<q{q}_root>/<tile_id>/RH{0..100}_Q{q}<file_suffix>`；Q1 是中位数，Q0/Q2 是区间下/上界。
- `q*_root` 相对于 `~/data/gvs/products/vsm`，就是版本 + 格式，数这一列就是每个版本的 tile 数：
  `awk -F'\t' 'NR>1 {print $3}' 2020/released_v0.tsv | sort | uniq -c`（Q1）。
- 同一个 tile 同一个 Q 的 101 个 RH 必须在同一个目录、目录名就是 tile_id、文件后缀一致。
  生成时逐个 tile 检查，有一个不满足就报错退出，不写 manifest（例如 retarget 只改了部分 RH，
  或 tile 目录还是旧的 `<TILE>_cog/` 命名）。
- `n_files` 是磁盘上实际存在的文件数（完整为 303），`size_bytes` 是它们的总大小；
  `n_files < 303` 的 tile 生成时会报 WARNING。

## 生成与更新

```bash
sbatch --job-name=release_manifest scripts/hendrix/run_cpu.sh release_manifest 2020
sbatch --job-name=release_manifest scripts/hendrix/run_cpu.sh release_manifest 2024
```

**2020 的 manifest 会过期**：每次 `stac_retarget run.dry_run=false` 把 href 从 geotiff/
改到 cog/ 之后都要重新生成。内容不变，变的是路径和大小。

## 下载

```bash
bash scripts/hendrix/download_tile_rclone.sh download_manifest 2020 32UPG 32UPF 33VUC 32UNG
```

落到 `<DEST_ROOT>/<year>/released_v0/<TILE>/`（一个 tile 一层），manifest 也拷一份到
`<DEST_ROOT>/<year>/released_v0.tsv`，作为本地文件来源的索引。
