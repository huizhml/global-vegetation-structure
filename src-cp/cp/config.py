import yaml


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


class EvalCPConfig:
    def __init__(self, config):
        self.data_root = config.get("data_root")
        self.data_path = config.get("data_path")


class GVSCPConfig:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.data = RHDataCPConfig(config["data"])
        self.cp = RunCPConfig(config["cp"])
        self.eval = EvalCPConfig(config["eval"])
