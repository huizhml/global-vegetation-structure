import os
import sys
import logging
from datetime import datetime

def setup_default_logging(log_path, string = 'Train', default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):

    output_dir = os.path.join(log_path)
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(string)

    def time_str(fmt=None):
        if fmt is None:
            fmt = '%Y-%m-%d_%H:%M:%S'
        return datetime.today().strftime(fmt)

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{time_str()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler(filename=os.path.join(output_dir, f'{time_str()}.log'), mode='a')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def df2gdf(df):
    """
    Convert a pandas.DataFrame to a geopandas.GeoDataFrame
    """
    import geopandas as gpd
    from shapely.geometry import Point
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[['.geo']]))

def plotPlygonPoints(pointdf, polygondf):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    polygondf.plot(ax=ax, facecolor="none", alpha=0.4, color='grey')
    pointdf.plot(ax=ax, color='red', markersize=1)
    plt.savefig('test.png')

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def get_class(name):
    class_module, class_name = name.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class