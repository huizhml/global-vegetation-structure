import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt

from .constants import BIOME_MAPPING, BIOME_SHORT_MAPPING

NATURE_FIG_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,  # To embed fonts in PDF for better compatibility
    "legend.fontsize": 5,
}


def set_plot_style(kwargs):
    mpl.rcParams.update(mpl.rcParamsDefault)
    for key, value in kwargs.items():
        plt.rcParams[key] = value


class RHDataCPConfig:
    class RHColumns:
        def __init__(self, rh_val, rh_cols: dict[str, str]):
            self.rh_val: int = rh_val
            self.ground_truth_col = rh_cols["ground_truth"]
            self.q_lo_col = rh_cols["q_lo"]
            self.q_med_col = rh_cols["q_med"]
            self.q_hi_col = rh_cols["q_hi"]

        def q_cols(self):
            return [self.q_lo_col, self.q_med_col, self.q_hi_col]

        def all_cols(self):
            return list(self.q_cols()) + [self.ground_truth_col]

    all_rh_cols: list[RHColumns]

    def __init__(self, config):
        self.biome_col = config["biome_col"]
        self.all_rh_cols = [
            self.RHColumns(rh_val, rh_cols)
            for rh_val, rh_cols in config["rh_cols"].items()
        ]
        self.all_rh_cols.sort(key=lambda x: x.rh_val)
        self.other_cols = config.get("other_cols", [])

    def __iter__(self):
        return iter(self.all_rh_cols)

    def get_all_rh_cols(self):
        all_rh_cols = [
            self.all_rh_cols[i].all_cols() for i in range(len(self.all_rh_cols))
        ]
        return [cols for rh_cols in all_rh_cols for cols in rh_cols]

    def get_all_cols(self):
        return self.other_cols + self.get_all_rh_cols()

    def get_rh_cols(self, rh_val: int):
        return next(
            (
                rh_cols.all_cols()
                for rh_cols in self.all_rh_cols
                if rh_cols.rh_val == rh_val
            ),
            None,
        )


class RunCPConfig:
    def __init__(self, config):
        self.alpha = float(config["alpha"])
        self.cqr_methods = config["cqr_methods"]
        self.cal_data_path = config.get("cal_data_path")
        self.cal_data_root = config.get("cal_data_root")
        self.preprocess_data = config.get("preprocess_data", True)


class EvalCPConfig:
    def __init__(self, config):
        self.data_root = config.get("data_root")
        self.data_path = config.get("data_path")
        self.preprocess_data = config.get("preprocess_data", True)


def pt_to_inch(pt_val):
    return pt_val / 72.27


class PlotConfig:
    def __init__(self, config):
        self.plot_style = NATURE_FIG_STYLE.copy()
        self.plot_style.update(config.get("plot_style", {}))
        self.fig_width_in = pt_to_inch(config.get("fig_width_pt", 511))
        self.fig_height_in = pt_to_inch(config.get("fig_height_pt", 324))
        self.flatten = config.get("flatten", False)
        self.legend_loc = config.get("legend_loc", "outside upper center")


class PlotCPConfig:
    def __init__(self, config):
        self.skip_biomes = [
            BIOME_SHORT_MAPPING[BIOME_MAPPING[b]]
            for b in config.get("skip_biomes", [])
        ]
        self._coverage = PlotConfig(config.get("coverage", {}))
        self._avg_width = PlotConfig(config.get("avg_width", {}))
        self.biome_order = [
            BIOME_SHORT_MAPPING[BIOME_MAPPING[i + 1]]
            for i in range(len(BIOME_MAPPING))
        ]

    def set_coverage_plot_style(self):
        set_plot_style(self._coverage.plot_style)

    def get_coverage_figsize(self):
        return self._coverage.fig_width_in, self._coverage.fig_height_in

    def get_avg_width_figsize(self):
        return self._avg_width.fig_width_in, self._avg_width.fig_height_in

    def set_avg_width_plot_style(self):
        set_plot_style(self._avg_width.plot_style)

    def get_avg_width_n_rows_cols(self):
        if self._avg_width.flatten:
            return 1, len(self.biome_order) - len(self.skip_biomes)
        else:
            return 2, 7

    def get_coverage_n_rows_cols(self):
        if self._coverage.flatten:
            return 1, len(self.biome_order) - len(self.skip_biomes)
        else:
            return 2, 7

    def get_coverage_biome_order(self):
        if self._coverage.flatten:
            return [b for b in self.biome_order if b not in self.skip_biomes]
        else:
            return self.biome_order

    def get_avg_width_biome_order(self):
        if self._avg_width.flatten:
            return [b for b in self.biome_order if b not in self.skip_biomes]
        else:
            return self.biome_order

    def get_coverage_legend_loc(self):
        return self._coverage.legend_loc

    def get_avg_width_legend_loc(self):
        return self._avg_width.legend_loc


class GVSCPConfig:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.data = RHDataCPConfig(config["data"])
        self.cp = RunCPConfig(config["cp"])
        self.eval = EvalCPConfig(config["eval"])
        self.plot = PlotCPConfig(config["plot"])
