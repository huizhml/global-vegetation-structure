# VSM 版本快照：snapshot

每个版本目录现在实际有什么。和 `released_v0.tsv` 分工不同：release manifest 记的是
**选了哪些文件**发布，snapshot 记的是**磁盘上现在有什么**，包括还在预测、转换中的年份。
版本内的路径是固定的（`tiles/<fmt>/<TILE>/RH*_Q*.tif`、`mosaic/...`），所以表里只记有什么，不记路径。

“版本”指年份下带 `tiles/` 或 `mosaic/` 的子目录（original、masked、blended、corrected…）；
`s5cmd_bench`、`predictions_2017` 这类目录跳过，运行时会打印出来。

三张表，都是普通 TSV，第一行就是表头：

| 文件 | 一行是 | 回答 |
|---|---|---|
| `<year>/snapshot_versions.tsv` | 一个版本 | 这一年每个版本下有什么 |
| `<year>/<version>/snapshot.tsv` | 一个 tile | 哪些 tile 有 cog / geotiff，各 Q 完整不完整 |
| `<year>/<version>/snapshot_mosaic.tsv` | mosaic 下一个文件 | mosaic 有哪些 RH、Q、文件类型 |

## snapshot_versions.tsv：年度总览

```
version         tiles  cog_tiles  cog_q0  cog_q1  cog_q2  geotiff_tiles  geotiff_q0  geotiff_q1  geotiff_q2  vrt_tiles  ...  mosaic_variables  mosaic_quantiles  mosaic_formats  ...  generated
bias_corrected  18143  18143      0       ...     0       0              0           0           0           19         ...                                                       ...  2026-10-17T06:17:42Z
masked          9251   9251       ...     ...     ...     0              0           0           0           0          ...  RH0-2,RH4-98      Q1                cog.tif,tif     ...  2026-10-17T06:18:05Z
```

- `<fmt>_tiles`：`tiles/<fmt>/` 下有这个 tile 的目录；`<fmt>_q<q>`：该 Q 的 101 个 RH 齐全的 tile 数。
- `mosaic_*`：mosaic 表对应列去重后的值，逗号分隔（不含 tmp 之类的目录）；连续的 RH 写成区间，
  `RH0-2,RH4-98` 就是缺 RH3、RH99、RH100。
- `generated`：这个版本被扫描的时间（UTC）。只扫部分版本时，只替换这些版本的行。

## snapshot.tsv：按 tile

```
tile_id  cog_q0  cog_q1  cog_q2  geotiff_q0  geotiff_q1  geotiff_q2  multiband  vrt  other_files  size_bytes
01GEL    101     101     101     0           101         0           0          0    0            ...
```

- `<fmt>_q<q>`：`tiles/<fmt>/<TILE>/` 里 `RH*_Q<q>.tif`（或 `.cog.tif`）的个数，**完整为 101**，0 就是没有。
  旧命名的 `<TILE>_cog/` 目录也算到 `<TILE>` 名下。
- `multiband`：`tiles/multibands/<TILE>_*.tif` 的个数。
- `vrt`：版本下或 `tiles/` 下任何 `vrt*` 目录里 `<TILE>*.vrt` 的个数
  （现有 `tiles/vrt/<TILE>_Q1.vrt`、`vrt/<TILE>_Q1.vrt`、`vrt_q1/<TILE>.vrt` 三种）。
- `other_files`：tile 目录里其他文件（临时文件、命名不对的），`.aux.xml` 不算。
- `size_bytes`：以上所有文件的总大小。
- 所有版本的列都一样，不同版本的表按 `tile_id` 直接 join。

```bash
# cog Q1 完整的 tile 数
awk -F'\t' 'NR>1 && $3==101' 2024/bias_corrected/snapshot.tsv | wc -l
# 有残缺或多余文件的 tile
awk -F'\t' 'NR>1 && ($3!=101 || $10>0)' 2024/bias_corrected/snapshot.tsv
```

## snapshot_mosaic.tsv：mosaic，按 RH

一个 RH 一行（还有 `CR`），每种 mosaic 文件一列，格子是文件数：

```
variable  global_mosaic.cog.tif  global_mosaic_Q0-Q2.cog.pdf  global_mosaic_Q1.cog.tif  global_mosaic_Q1.tif  tmp_tiles_resampled_1km_Q1/  vrt_tmp/  n_files  size_bytes  modified
RH25      0                      1                            1                         1                     18143                        0         18146    ...
RH98      0                      1                            1                         1                     18143                        0         18146    ...
CR        1                      0                            0                         0                     0                            0         1        ...
other     0                      0                            0                         0                     0                            12        13       ...
```

- 文件名拆成 `<kind>[_<year>]_<variable>[_<quantile>].<format>`，`variable` 是 `RH<n>` 或 `CR`，
  列名是 `<kind>[_<quantile>].<format>`；`mosaic/cog/`、`mosaic/geotiff/` 里的文件列名带 `cog/`、`geotiff/` 前缀。
- 以 `/` 结尾的列是 `mosaic/` 下的其他子目录（`tmp_tiles_resampled_*` 之类），格子是目录里的文件数，方便找出可以清理的中间结果。
- 拆不开的名字（`README.md`、`run.log`、`vrt_tmp/`）各自一列，放在最后一行 `other`。
- `n_files`、`size_bytes`、`modified`（最新修改时间，UTC）是整行的合计。
- 列随版本里实际有的文件而定，不同版本的列不一样；没出现的 RH 就没有这一行，
  缺哪些 RH 看总览里的 `mosaic_variables`（如 `RH0-2,RH4-98`）。

## 生成与更新

```bash
conda activate inference && cd ~/gvs
sbatch --job-name=vsm_snapshot scripts/hendrix/run_cpu.sh vsm_snapshot 2024
# 只做某几个版本
python -m tools.run run=vsm_snapshot run.year=2020 'run.versions=[masked,blended]'
```

工具只统计、不校验：认不出的目录或文件会在日志里列成 `NOTE`，不会报错退出。
快照会过期，预测、转换、`stac_retarget` 改动某个版本之后要重新生成。
