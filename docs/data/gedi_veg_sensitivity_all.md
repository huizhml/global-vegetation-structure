# GEDI L2A：veg_sensitivity_all

按 Sentinel-2 tile 切分、每年一份的 GEDI L2A 有效 footprint，**不做 sensitivity 过滤**。
`veg_sensitivity_gt0p95` 的数据都可以由它在本地筛 `sensitivity >= 0.95` 得到，
以后的年份只下载这一份。

本文件的源文件：仓库里的 `docs/data/gedi_veg_sensitivity_all.md`（数据目录下的 `README.md` 是它的拷贝）。

## 位置与目录结构

Hendrix：`~/data/gvs/gedi/veg_sensitivity_all/`（即 `/projects/dereeco/data/gvs/gedi/veg_sensitivity_all/`）
ERDA 备份：`ucph-erda:GEDI/veg_sensitivity_all/<year>/`（`download_gedi.sh upload_erda`）

```
<year>/<tile>.parquet        每个 S2 tile 一个文件（点 + 全部 L2A 属性 + mgrs_tile）
<year>_state/                下载进度（tools/task_state.py）：每个 tile 一行 done/failed
                             + 说明；expected.tsv、uploads.tsv、mgrs_summary_*.tsv
orbit_overview/              每个 tile 每年的 orbit 总览（见下文）
```

## 抽样规则（当前版本，2026-10-15 起）

| 项目 | 规则 |
|---|---|
| 数据源 | GEE `LARSE/GEDI/GEDI02_A_002`，每个 granule（约 1/4 条轨道）一张表 |
| 选哪些 orbit | 当年 `LARSE/GEDI/GEDI02_A_002_INDEX` 中与 tile 多边形相交、且月份在该 tile `growing_months` 里的表（`orbit_months: growing`）。**生长月份内一条 orbit 都没有的 tile，改用全年的 orbit** |
| 跨 180° 的 granule | index 里约 8% 的 granule 经度连续写到 180 以外；求交时把 tile 平移 ±360° 再比一次（`tile_orbit_pairs`），2026-10-15 之前没有这一步，见“已知问题” |
| 空间范围 | 每个 S2 tile 用自己的多边形 `filterBounds`；S2 grid 有 18156 个 tile，其中 25 个跨 180°，两部分分别查询 |
| 逐点过滤 | `quality_flag == 1`、`degrade_flag == 0`、`region_class > 0`、`leaf_off_flag != 1`、`pft_class ∈ 1..8` |
| 水体过滤 | `digital_elevation_model > -1000`（没有 DEM 的点基本是海面）且 `landsat_water_persistence < 50`（湖面、河面）。2026-10-15 起在下载时过滤，之前下载的文件用 `drop_water` 任务在本地补做 |
| sensitivity | 不过滤（`sensitivity_threshold: null`），需要时在本地筛 |
| 抽样 | 不抽样，保留全部满足条件的点 |
| 重复 | S2 tile 之间有重叠，重叠区的点会出现在多个 tile 文件里；`mgrs_tile == tile` 可以让每个点只保留一次 |
| 下载方式 | GEE 导出 CSV；`shot_number` 以 int64 读入，已核对没有被舍入（2026-10-15，抽查 8 个 tile） |

growing months 来自 `~/data/gvs/01_derived_ancillary/growing_months/s2_tiles_with_growing_months.parquet`（由 VNP22Q2 2020 推出，见 `download/products/growing_season.py`）。落叶期的点再由 `leaf_off_flag != 1` 逐点去掉；L2A 的说明是这个字段来自 VIIRS 物候，常绿区和没有物候数据的地方是 255，不会被去掉。

### 为什么按生长季、为什么暂不更新 growing months

- **数据量已经足够。** 2020 年按生长季下载的约有 14.8 亿个点，训练抽样每年只需约 7500 万个。改成全年会多出 50–130%（试跑：10UEB 从 49 万涨到 114 万），GEE 配额、时间和存储都不划算。
- **和产品的季节一致。** 预测用的是生长季的 S2 合成影像，验证用的 GEDI 点也取同样的月份。
- **growing months 暂不更新。** S2 合成影像用的也是这一版月份，GEDI 单独换一版，验证和产品的季节就对不上；补漏也要靠这一版月份判断哪些 orbit 已经下过。诊断显示规则基本合理（见“诊断”），问题主要是常绿区月份偏少、干旱区和北极用了“取前 5 个月”的兜底，影响的是数据量，不是对错。以后要改（例如用 MCD12Q2 多年气候态），应和下一版产品一起换。GEDI 不必重下：由 `delta_time` 可以算出月份在本地重新筛，新月份多出来的部分用同样的补漏方式补下。

### 补漏（2026-10-15）

2019–2022、2024 已按旧规则下载过。用新代码再跑一次同一年，会**原地补漏**：每个 tile 只下载之前漏掉的 orbit，即跨 180° 的 granule，以及生长月份内没有 orbit 的 tile 的全年 orbit，追加进原 parquet（按 `shot_number` 去重），原有的行也按同样的水体条件过滤。没有漏掉 orbit 的 tile 不重写，只在 state 里记为完成，所以之后还要跑一次 `drop_water`。前提是 S2 grid 文件和 orbit index 与第一次下载时相同。

```bash
sbatch scripts/hendrix/download_gedi.sh download year=2020        # 补漏 / 新下载
sbatch scripts/hendrix/download_gedi.sh drop_water year=2020      # 补漏完成后，去掉原有文件里水上的点
```

state 中补漏完成的 tile，说明以 `growing months:` 开头，例如 `growing months: 5321 shots (+830 from 12 orbits)`、`growing months: 4 shots (nothing missing)`；用了全年兜底的 tile 会注明 `no orbit in growing months: all months`。如果出现 `N already there`，说明有 orbit 被重复下载，需要检查 S2 grid 或 orbit index 是否和第一次下载时不同。所有 tile 都是 `done` 且说明以 `growing months:` 开头，这一年才算完整（`state_summary` 显示 COMPLETE 不够，旧的记录也算 done）。

`orbit_months: all`（全年所有 orbit）的代码也保留着，以后需要时同一年可以再补成全年，说明以 `all months:` 开头。

## 与训练数据（GEDI_high_sens → h5 → gt0p95 subsets）的对比

训练数据由旧版 `GEDI.download`（2024 年 1–2 月，`download/products/gedi.py` 中的 `download_zone`/`download_orbit`）下载，
存为 `modeling/gedi/GEDI_high_sens/<year>/<zone>/<orbit>.parquet`，再经
`GEDI_extra* → s2_download → GEDI_S2_h5_extra → preprocessing step1–3 → extract_gedi_from_h5`
得到 `veg_sensitivity_gt0p95/subset_*`。

| | veg_sensitivity_all（当前） | 训练数据（GEDI_high_sens） |
|---|---|---|
| 质量过滤 | quality, degrade, region_class, leaf_off_flag != 1 | 相同 |
| 水体 | 去掉没有 DEM 和 Landsat 水体频率 >= 50% 的点 | 不过滤（训练时用 WorldCover 永久水体掩膜排除） |
| sensitivity | 不过滤 | `>= 0.95` |
| pft_class | 只保留 1–8 | 不过滤 |
| 生长季 | tile growing months（没有生长月份 orbit 的 tile 用全年） | 无 |
| 空间单位 | S2 tile 多边形 | MGRS grid zone（如 `32T`）的 bbox |
| 抽样 | 全部 | 按陆地面积随机抽样，约 0.70 点/km²（全球约 7500 万/年），GEE `randomColumn(seed=42)` |
| 年份 | 2019– | 2019–2022 |

要和训练数据对齐，需要在本地筛 `sensitivity >= 0.95`。

### 和训练数据不同的地方，以及原因

- **sensitivity 不过滤（训练用 `>= 0.95`）。** 后面的验证需要全部 sensitivity 的点，例如看模型在低 sensitivity 点上的表现，或者 sensitivity 阈值对评估结果的影响。所以下载时不过滤，要和训练数据对齐时在本地筛。
- **只保留 `pft_class` 1–8（训练下载时不过滤）。** 训练时非植被点本来就不参与 RH 回归：`filter_out_nonveg: True`，见 `config/train.yaml` 和 `models/vs_regression.py`。对冠层高度、结构这类任务，没必要下载非植被点的 RH profile。
  - 两者对“非植被”的定义并不相同，pft 1–8 更严。
  - 训练用的是 h5 patch 中心像元的 **ESA WorldCover**（`models/modules/util.py: get_veg_mask`），只排除 Built-up、Snow and ice、Permanent water bodies 三类。
  - 下载用的是 GEDI 自带的 `pft_class`（MODIS PFT）。保留 1–8，去掉 0 water、9 urban、10 snow/ice、11 barren。
  - 所以 pft = 11（barren）的点不在本数据里，而训练时 WorldCover 为 Bare/sparse vegetation 的点仍会参与 RH 回归。
- **按生长季（训练不按）。** 见上文“为什么按生长季”：和产品季节一致，数据量也足够。
- **去掉水上的点（训练下载时不过滤）。** 没有 DEM 的点基本是海面，冠层高度只有约 3 m，`pft_class` 却常被标成森林，留着会把验证结果往低拉。训练时这类点由 WorldCover 永久水体掩膜排除，效果相近。
- **不抽样（训练按陆地面积随机抽，约 0.70 点/km²）。** 本数据要作为所有任务共用的底表，保留全部点；需要和训练相同的密度时，在本地抽样。

注意：`veg_sensitivity_gt0p95/subset_*` 由 `extract_gedi_from_h5` 生成，**shot_number 被 float64 舍入过**
（2026-10-15 已修代码，旧数据未修复）。不能单靠 shot_number 和本数据 join，要按位置匹配。

## 各年状态

截至 2026-10-15，以 `state_summary` 为准：

| 年份 | 状态 |
|---|---|
| 2019 | 最初用了旧的 orbit index 缓存（790 KB，缺大部分 orbit），之后用完整 index 按 growing 规则整年重下；待补漏 |
| 2020 | growing 规则下载完成（18156 个 tile 全部 done），待补漏 |
| 2021 | growing 规则下载完成，待补漏；部分已传到 ERDA 的文件缺 `mgrs_tile` 列 |
| 2022 | 13893 个 tile 失败（GEE Restricted Mode），用新规则重跑即可 |
| 2023– | 未下载 |

**跨 180° 经线的 granule 被漏掉（2026-10-15 发现并修复）。** GEE orbit index 里约 8% 的 granule 跨过 180° 后坐标连续写下去（2019–2024 每年 999–1836 个 `lon > 180`，最远写到 263.4°，即西经 96.6°；没有 `< -180`）。旧的求交只用原坐标，这些 granule 匹配不到它们实际经过的西半球 tile。2020 年共有 183 个 tile 漏掉了一部分 orbit：

| 区域 | 例子 | 漏掉的 orbit 比例 |
|---|---|---|
| 斐济、汤加、克马德克群岛（经度 -179 到 -174） | 01KB*、01KC*、01LCE | 大多 100%，这些 tile 之前被记为“没有 orbit 经过” |
| 阿留申群岛、萨摩亚、夏威夷、法属波利尼西亚、马克萨斯群岛（经度 -179 到 -138） | 01U*、02L*、04Q*、05K*、07L* | 30–90%（夏威夷多为 60–90%） |
| 加拿大西部 / 美国西北，北纬 48–52°（经度 -131 到 -97） | 09U*–14U* | 10–24%（北纬 51.6° 附近轨道近乎东西走向，一个 granule 能横跨 100 多度经度） |
| 跨 180° 的 25 个 tile 的西侧那一半 | 01KAB 等 | 几乎 100%（各年有文件的 9–11 个，西侧全是 0 个点） |

现在 `tile_orbit_pairs` 会把 tile 平移 ±360° 再求交一次。补漏时，平移后才匹配上、且月份在生长月份内的 orbit（`wrapped`）会被补下，上面这些 tile 会被自动补齐。跨 180° 那几个 tile 的旧文件已挪到 `antimeridian_half_backup/<year>/`，state 也已 reset（2019–2021 已做，2022、2024 待做），补漏时会两半分别重下。

## Orbit 总览

```bash
python -m tools.run run=gedi_orbit_overview      # -> orbit_overview/
```

- `tile_orbits_<year>.parquet`：每个（tile, orbit 表）一行，包括 `table_id`、`orbit`、`month`、`in_growing`。
- `tile_orbit_counts.tsv`：每个 tile 一行，包括 `orbits_<y>`、`orbits_growing_<y>`、`few_orbits_<y>`，以及下载 state 里的 `shots_<y>`、`download_<y>`。
- `orbit_summary.tsv`：每年一行。

`few_orbits_<y>` 标记 orbit 数少于同一 8° 纬度带中位数 40% 的 tile，可能是下面“已知问题”里的 orbit index 异常。

## 诊断（2020，2026-10-15）

工具：`python -m tools.run run=growing_season_audit run.year=2020`
（输出在 `~/data/gvs/01_derived_ancillary/growing_months/growing_months_audit/2020/`，和 growing months parquet 放在一起，因为 deploy 筛 S2 影像也用它）。

**growing months 由哪条规则得出**（共 18156 个 tile）：

| 规则 | tile 数 | 分布 |
|---|---|---|
| 超过 50% 像元处在生长季的月份 ≥ 5 个 | 14433 | 大部分陆地 |
| 不足 5 个月，取频率最高的 5 个月 | 2545 | 北极、萨赫勒、澳洲中部、青藏高原、巴塔哥尼亚 |
| 无物候像元，用默认值（北 5–9 月 / 南 12–5 月） | 1178 | 撒哈拉/阿拉伯、北极 |

热带基本都走第一条规则，没有被套用默认月份。

**下载结果 × orbit 情况：**

| | 生长月份内有 orbit | 完全没有 orbit 经过 | orbit 都不在生长月份内 |
|---|---|---|---|
| N shots | 12538 | 0 | 0 |
| no orbit in growing season | 0 | 4137 | 97 |
| no valid shots | 1384 | 0 | 0 |

- 4137 个 tile 基本都在 GEDI 覆盖范围（±51.6°）以外，与生长季无关。现在的代码会把这类 tile 记为 `no GEDI orbit over tile`。
- 1384 个 tile 在生长月份里有 orbit，但过滤后一个点都不剩，推测多为荒漠，尚未逐一核对。
- 97 个 tile 的情况见“已知问题”。

**按 growing months 筛选时保留下来的 orbit 比例**（|lat| < 51.6°，每个 tile 取比例后按 WWF BIOME 求中位数）：

| BIOME | 类型 | 生长月份数（中位数） | 保留的 orbit 比例 |
|---|---|---|---|
| 1 | 热带/亚热带湿润阔叶林 | 10 | 0.87 |
| 2 | 热带/亚热带干燥阔叶林 | 9 | 0.82 |
| 4 | 温带阔叶混交林 | 8 | 0.68 |
| 5 | 温带针叶林 | 7 | 0.63 |
| 6 | 北方针叶林 / 泰加林 | 6 | 0.50 |
| 7 | 热带/亚热带草原、稀树草原 | 9 | 0.73 |
| 8 | 温带草原 | 7 | 0.57 |
| 12 | 地中海林地 | 7 | 0.64 |
| 13 | 荒漠与干旱灌丛 | 5 | 0.51 |

## 已知问题

- **97 个 tile 的 orbit 数异常少。** 2020 年这 97 个 tile 每个只有 12–18 条 orbit，同纬度带其他 tile 有 50–90 条，而且这些 orbit 几乎都在 6–7 月。怀疑是 orbit index 的问题：要么这些 tile 的 grid 几何有问题，要么某些 granule 的几何异常。**尚未查明。**
  - 其中太平洋的几个 tile（01LHC、02LKH 等）属于跨 180° 漏掉 orbit 的情况。其余大多在亚马孙、萨赫勒、澳洲内陆，原因未查明；补漏时这些 tile 会改用全年的 orbit，至少能拿到这十几条 orbit 的点。
  - `orbit_overview` 里的 `few_orbits_<y>` 列会标出这些 tile。
- **`veg_sensitivity_gt0p95/all_valid/2020` 是本数据的子集，可以删除。** 抽查 8 个可读的 tile，点全部能以相同 id、相同位置在本数据中找到。另外抽查的 20 个文件里有 12 个读不出来（errno 5，存储层问题，需要报告给 Hendrix 管理员）。删除前需要把 `extract_lvis_gedi_pairs`、`add_slope` 两个配置改为读本数据。

## 相关代码

- 下载：`download/products/gedi.py`（`GEDI.download_all_valid`、`tile_orbit_pairs`），配置项 `config/download/config.yaml: download_all_valid_gedi`，作业脚本 `scripts/hendrix/download_gedi.sh`
- 完整性检查：`tools/gedi_coverage.py`，以及 `run=state_summary`
- 诊断：`tools/growing_season_audit.py`、`tools/gedi_orbit_overview.py`、`tools/gedi_extract_cols.py`（`gedi_diagnose_keys`）
