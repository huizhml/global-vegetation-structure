# VSM: A Vertical Vegetation Structure Model of the Earth

<p align="center">
  <a href="https://huizhml.github.io/map-explorer/">
    <img src="https://raw.githubusercontent.com/huizhml/global-vegetation-structure/master/docs/assets/vsm_global_every2rhs_black_bg_v7.png" alt="Global map of GVSM vertical vegetation structure, with RH layers stacked as a 3-D datacube (height in metres, 0–50)" width="100%">
  </a>
</p>

VSM brings a vertical dimension to global vegetation mapping, extending beyond canopy height to characterzing vegetation structure at 10 m resolution. 
Trained with 178 million sparse [GEDI](https://gedi.umd.edu/) lidar observations, a deep learning model translates Sentinel-2 imagery into continuous maps of 101 relative height metrics (RH0–RH100), each with predictive quantiles that describe its uncertainty.
By capturing vegetation structure beyond canopy height, VSM supports applications in forest monitoring, biodiversity assessment, and ecosystem research.

## Explore

- 🗺️ **Interactive map:** [huizhml.github.io/map-explorer](https://huizhml.github.io/map-explorer/)
- 💻 **Code:** [github.com/huizhml/global-vegetation-structure](https://github.com/huizhml/global-vegetation-structure) 
- 📦 **Storage:** `s3://geoai-ucph/gvsm/` on the endpoint `https://data.source.coop`

## Key facts

| | |
|---|---|
| Variable | Relative height profile (RH0 - RH100) , in decimeters |
| Uncertainty | Median (Q1) available,  $5^{th}$ percentile(Q2) and $95^{th}$ percentile(Q0) coming soon |
| Resolution | 10 m (main product), 1 km (global mosaics) |
| Coverage | Global land |
| Years | 2020 (available), 2024 (in preparation) |
| Input | Sentinel-2 L2A |
| Reference data | GEDI L2A, 178 million height profiles |
| Format | Cloud Optimized GeoTIFF, Int16, lossless LERC+ZSTD (tiles) / ZSTD (mosaics) |
| Volume | ≈ 360 TB per year (full RH profile × 3 quantiles) |
| License | CC BY 4.0 |

## What is an RH metric?

GEDI records laser energy reflected by vegetation and the ground within each footprint. **RHn** is the height above ground below which *n* % of that returned energy has accumulated.

- **RH98–RH100**: approximate canopy top height, comparable to conventional canopy height products.
- **RH50**: the height below which half of the total returned energy has accumulated.
- **Low RHs (RH0–RH25)**: describe the lower portion of the cumulative return, influenced by both vegetation and ground returns.

Taken together, RH0 … RH100 for a pixel make a cumulative vertical profile. You can use it to study canopy layering, understory density and structural diversity, not only the top height.

<img src="https://raw.githubusercontent.com/huizhml/global-vegetation-structure/master/docs/assets/1a_vertical_structure.png" alt="Global map of VSM vertical vegetation structure, with RH layers stacked as a 3-D datacube (height in metres, 0–50)" width="100%">

## Data organization

```
gvsm/
├── <year>/                      # acquisition year, e.g. 2020
│   └── <MGRS tile>/             # Sentinel-2 MGRS tile ID, e.g. 32UNG
│       ├── RH0_Q1.tif
│       ├── RH1_Q1.tif
│       ├── …
│       └── RH100_Q1.tif
└── mosaics/
    └── <year>/
        ├── RH0_Q1.tif
        ├── …
        └── RH100_Q1.tif
```

File names follow `RH{level}_Q{quantile}.tif`, with `level` from 0 to 100 and `quantile` from 0 to 2.

| Path | Content | Grid |
|---|---|---|
| `<year>/<tile>/` | Full-resolution predictions. One folder per Sentinel-2 MGRS tile, one file per RH level and quantile. | Native UTM zone of the tile, 10 m, 10 980 × 10 980 px |
| `mosaics/<year>/` | Global overview mosaics (10 m tiles averaged to 0.01°), for browsing and continental- to global-scale analysis | EPSG:4326, 0.01°, 36 000 × 13 889 px |

To find the tile that covers your area, use the [Sentinel-2 MGRS tiling grid](https://sentiwiki.copernicus.eu/web/s2-products). Neighbouring tiles overlap by about 10 km.

## Quantiles: Q0, Q1, Q2

The model is trained with quantile regression, so each RH level is predicted at three quantiles:

| Suffix | Quantile | Use |
|---|---|---|
| `Q1` | 50th percentile (median) | **The main prediction.** Use this by default. |
| `Q2` | 5th percentile | Lower bound of the prediction interval |
| `Q0` | 95th percentile | Upper bound of the prediction interval |

`Q0` and `Q2` together give a **90 % prediction interval** for each pixel and RH level. The interval is calibrated with conformal prediction, so it reaches the target coverage against held-out GEDI data. The interval width (`Q0 − Q2`) is a simple per-pixel uncertainty measure.

> **Note:** Only `Q1` (the median) is published at the moment. `Q0` and `Q2` will be uploaded to the same folders soon.

## Pixel values

- Data type `Int16`. **Values are in decimetres**: divide by 10 to get metres.
- NoData = `32767`. This marks pixels without a valid prediction, such as water, snow and ice, persistent cloud.
- Low RH levels can be **negative**. This follows the GEDI convention: the lowest part of the return waveform lies below the detected ground elevation.

## Quick start

**GDAL**: inspect a file over HTTP without downloading it:

```bash
gdalinfo /vsicurl/https://data.source.coop/geoai-ucph/gvsm/2020/32UNG/RH98_Q1.tif
```

**Python**: read the vertical profile at a point:

```python
import numpy as np
import rasterio
from pyproj import Transformer

base = "https://data.source.coop/geoai-ucph/gvsm/2020/32UNG"
lon, lat = 9.7915, 55.4934

profile = []
for rh in range(101):
    with rasterio.open(f"{base}/RH{rh}_Q1.tif") as src:
        x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
        v = next(src.sample([(x, y)]))[0]
        profile.append(np.nan if v == src.nodata else v / 10)  # decimetres -> metres
```

**AWS CLI**: list or download data:

```bash
aws s3 ls --no-sign-request --endpoint-url https://data.source.coop s3://geoai-ucph/gvsm/2020/32UNG/
aws s3 sync --no-sign-request --endpoint-url https://data.source.coop s3://geoai-ucph/gvsm/mosaics/2020/ ./gvsm_mosaics_2020/
```

The COGs also open directly in QGIS (*Layer → Add Raster Layer → Protocol: HTTP(S)*) and in `rioxarray`/`xarray`.

## Caveats

- **These are model predictions, not lidar measurements.** Accuracy varies with biome, terrain slope and canopy density. In dense tall forests, predicted heights tend to be biased toward the mean.
- **GEDI only samples between about 51.6° N and 51.6° S.** Predictions further north and south rely on the model extrapolating beyond its training data, so treat them with extra caution.
- **RH metrics are not the same as airborne-lidar canopy height.** GEDI's RH metrics come from a ~25 m footprint waveform. They are related to, but not identical to, canopy height models derived from airborne laser scanning (ALS).
- Each year's map is built from Sentinel-2 imagery of that year. Differences between years mix real change with differences in the input imagery, so change analysis needs care.
- The global mosaics are 10 m predictions averaged to ~1 km. They smooth out fine-scale structure, so use the 10 m tiles for quantitative work.

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

## Citation

*Paper in preparation. Citation will be added here.*

## Contact

Managed by GeoAI@UCPH, [University of Copenhagen](https://www.ku.dk).

Hui Zhang (huzh@di.ku.dk) · Nico Lang (nila@di.ku.dk) · Christian Igel (igel@di.ku.dk)
