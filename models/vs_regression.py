import torchmetrics
import torch
import torch.nn as nn
import numpy as np
from typing import Any
from lightning import LightningModule
from kornia.enhance import normalize
from pathlib import Path
from models.metrics import MAE, RMSE, ME
from models.modules.util import get_veg_mask
from const import LAT_MEAN, LAT_STD, LON_SIN_MEAN, LON_SIN_STD, LON_COS_MEAN, LON_COS_STD, SLOPE_MEAN, SLOPE_STD

MASKED_VALUE = 32767
S2_MEAN = np.array([587.0255, 1361.4782, 1091.5225,  739.6226, 1772.4885, 2550.3611,
        2870.9454, 2952.2076, 3080.6981, 3086.7147, 2886.7224, 2175.1265
    # 630.6893, # 628.1879 , # unfiltered
    #              1359.3749, # 1475.1014 , 
    #              1108.1755, # 1165.6053 , 
    #              747.1477, # 794.6629 , 
    #              1778.6479,  # 1873.0237 , 
    #              2570.6654, # 2563.7012 , 
    #              2872.9004, # 2852.5837 , 
    #              2942.0968, # 2928.7439 , 
    #              3083.4950, # 3045.0754 , 
    #              3091.6063, # 3052.8425 , 
    #              2837.6595, # 3006.8704 , 
    #              2107.9847, # 2330.9114 ,
        ])
S2_STD = np.array([529.5716, 1314.4954,  846.1677,  635.8849, 1293.3423, 1078.8700,
        1119.9127, 1133.3539, 1116.3337, 1100.1368, 1651.3810, 1718.6127 # train1_v3
    # 608.2745, 1320.2338,  875.1261,  689.7431, 1298.4036, 1109.1152,
        # 1152.3742, 1167.4110, 1160.0978, 1145.2506, 1639.9369, 1669.9939 # 1698.3635
        # 20.39255142211914,
        # 0.385648638010025,
        # 0.3675101101398468
        ])

class VSRegression(LightningModule):

    def __init__(
            self,
            backbone: nn.Module = None,
            augment: nn.Module = None,
            feed_slope: bool = False,
            zero_slope: bool = False,
            slope_th: float = 20,
            zero_out_nonveg: bool = False,
            filter_out_nonveg: bool = False,
            feed_latlon: bool = False,
            evaluate_high_slope: bool = False,
            loss_fc: nn.Module = None,
            transform: nn.Module = None,
            yhat_transform: nn.Module = None, 
            stats_dir: str = None,
            load_from_file: bool = False,
            *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.augment = augment
        # self.backbone = backbone
        if backbone is not None:
            for name, module in backbone.named_children():
                setattr(self, name, module)
        self.forward = backbone.forward
        
        self.feed_slope = feed_slope
        self.zero_slope = zero_slope
        self.slope_th = slope_th
        self.feed_latlon = feed_latlon
        self.zero_out_nonveg = zero_out_nonveg
        self.filter_out_nonveg = filter_out_nonveg
        self.evaluate_high_slope = evaluate_high_slope
        self.loss_fc = loss_fc
        self.transform = transform
        self.yhat_transform = yhat_transform
        if load_from_file:
            stats_dir = Path(stats_dir).expanduser()
            self.lat_mean = np.loadtxt(stats_dir / 'lat_mean_filtered.txt')
            self.lat_std = np.loadtxt(stats_dir / 'lat_std_filtered.txt')
            self.lon_sin_mean = np.loadtxt(stats_dir / 'lon_sin_mean_filtered.txt')
            self.lon_sin_std = np.loadtxt(stats_dir / 'lon_sin_std_filtered.txt')
            self.lon_cos_mean = np.loadtxt(stats_dir / 'lon_cos_mean_filtered.txt')
            self.lon_cos_std = np.loadtxt(stats_dir / 'lon_cos_std_filtered.txt')
            self.slope_mean = np.loadtxt(stats_dir / 'slope_mean_filtered.txt')
            self.slope_std = np.loadtxt(stats_dir / 'slope_std_filtered.txt')
            self.s2_mean = np.loadtxt(stats_dir / 's2_mean_filtered.txt')
            self.s2_std = np.loadtxt(stats_dir / 's2_std_filtered.txt')
        else:
            self.lat_mean = LAT_MEAN
            self.lat_std = LAT_STD
            self.lon_sin_mean = LON_SIN_MEAN
            self.lon_sin_std = LON_SIN_STD
            self.lon_cos_mean = LON_COS_MEAN
            self.lon_cos_std = LON_COS_STD
            self.slope_mean = SLOPE_MEAN
            self.slope_std = SLOPE_STD
            self.s2_mean = S2_MEAN
            self.s2_std = S2_STD
        self.train_metrics = torchmetrics.MetricCollection(
            {
                'MAE/': MAE(),
                'RMSE/': RMSE(),
                'ME/': ME(),
                **{f'MAE_h>{i}/': MAE(threshold=i) for i in [30, 40, 50]},
                **{f'RMSE_h>{i}/': RMSE(threshold=i) for i in [30, 40, 50]},
                **{f'ME_h>{i}/': ME(threshold=i) for i in [30, 40, 50]}
                # # 'mape': MAPE()
            }, compute_groups=False,  # update process is different for these metrics
            postfix='--- train'
        )
        self.val_dataname = 'val'
        self.val_metrics = self.train_metrics.clone(postfix=f'--- {self.val_dataname}')
        self.val_metrics_veg = self.train_metrics.clone(postfix=f'--- {self.val_dataname} --- veg')
        self.val_metrics_lcc = torchmetrics.Accuracy('multiclass', num_classes=12, multidim_average='global')
        if self.evaluate_high_slope:  # for bias correction
            self.val_me_with_steep_slope = ME()
        self.save_hyperparameters()

        # Predefine processing functions based on flags
        if self.filter_out_nonveg and self.zero_out_nonveg:
            raise ValueError('filter_out_nonveg and zero_out_nonveg cannot be both True')
        self.process_slope = self._process_slope if self.feed_slope else lambda x, slope_arr: x
        self.process_latlon = self._process_latlon if self.feed_latlon else lambda x, lat, lon: x
        self.process_nonveg = self._process_nonveg if self.zero_out_nonveg else lambda rhs, veg_mask: rhs
        self.add_veg_mask = self._add_veg_mask if self.filter_out_nonveg else lambda veg_mask, slope_mask: slope_mask
        self.process_slope_evaluation = self._process_slope_evaluation if self.evaluate_high_slope else lambda rhs, rhs_hat, lc, lat, lon, mask: (rhs[mask], rhs_hat[mask], lc[mask], lat[mask], lon[mask])

    
    def _process_slope_evaluation(self, rhs, rhs_hat, lc, lat, lon, mask):
        self.val_me_with_steep_slope(rhs_hat, rhs)
        return rhs, rhs_hat, lc, lat, lon

    def get_slope_mask(self, slope):
        loss_mask_slope = torch.where(slope < self.slope_th, 1, 0)
        return loss_mask_slope.type(torch.bool)

    def _process_slope(self, x, slope_arr):
        if self.zero_slope:
            slope = torch.zeros_like(slope_arr)
        else:
            slope = torch.nan_to_num(slope_arr, nan=0)
        slope = torch.nan_to_num(slope_arr, nan=0)
        slope = (slope - self.slope_mean) / self.slope_std
        return torch.cat([x, slope.unsqueeze(1)], dim=1)

    def _process_latlon(self, x, lon, lat):
        patch_size = lon.shape[1]
        lon = lon.unsqueeze(1).repeat(1, patch_size, 1)
        lat = lat.unsqueeze(2).repeat(1, 1, patch_size)
        sin_lon = torch.sin(lon * torch.pi / 180)
        cos_lon = torch.cos(lon * torch.pi / 180)
        lat = (lat - self.lat_mean) / self.lat_std
        sin_lon = (sin_lon - self.lon_sin_mean) / self.lon_sin_std
        cos_lon = (cos_lon - self.lon_cos_mean) / self.lon_cos_std
        return torch.cat([x, lat.unsqueeze(1), sin_lon.unsqueeze(1), cos_lon.unsqueeze(1)], dim=1)

    def _process_nonveg(self, rhs, veg_mask):
        return rhs * veg_mask.unsqueeze(-1)

    def _add_veg_mask(self, veg_mask, slope_mask):
        return (veg_mask & slope_mask).bool()

    
    # def on_fit_start(self):

    #     self.quantize_model()
    #     import ipdb; ipdb.set_trace()
    #     return super().on_fit_start()
    
    def on_train_start(self):
        self.s2_mean = torch.tensor(self.s2_mean).to(self.device)
        self.s2_std = torch.tensor(self.s2_std).to(self.device)
        self.lat_mean = torch.tensor(self.lat_mean).to(self.device)
        self.lat_std = torch.tensor(self.lat_std).to(self.device)
        self.lon_sin_mean = torch.tensor(self.lon_sin_mean).to(self.device)
        self.lon_sin_std = torch.tensor(self.lon_sin_std).to(self.device)
        self.lon_cos_mean = torch.tensor(self.lon_cos_mean).to(self.device)
        self.lon_cos_std = torch.tensor(self.lon_cos_std).to(self.device)
        self.slope_mean = torch.tensor(self.slope_mean).to(self.device)
        self.slope_std = torch.tensor(self.slope_std).to(self.device)
        return super().on_train_start()
    
    def on_train_epoch_start(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.val_metrics_veg.reset()
        self.val_metrics_lcc.reset()
        if self.evaluate_high_slope:
            self.val_me_with_steep_slope.reset()
        return super().on_train_epoch_start()
    
    
    def training_step(self, sample, batch_idx):
        x = self.augment(sample[0].float())
        x = normalize(x, self.s2_mean, self.s2_std)
        # x = self.transform(x)
        rhs = sample[1]
        slope_mask = self.get_slope_mask(sample[3][..., 7, 7])
        veg_mask = get_veg_mask(sample[2]) # NOTE: make all masks based on the raw batch size
        # NOTE: veg mask will filter out nonveg samples if filter_out_nonveg is True
        mask = self.add_veg_mask(veg_mask, slope_mask)
        # NOTE: this step will zero out nonveg samples if zero_out_nonveg is True
        rhs = self.process_nonveg(rhs, veg_mask)
        # NOTE: this step will add slope as input if feed_slope is True, sample[3] is the slope array
        x = self.process_slope(x, sample[3])
        # NOTE: this step will add lat and lon as input if feed_latlon is True, sample[4] and sample[5] are the lat and lon vectors
        x = self.process_latlon(x, sample[4], sample[5])
        y_hat = self.forward(x.float())  # y_hat is (n, 303, 15, 15)

        losses, pred, target = self.loss_fc(y_hat, mask, rhs, *sample[2:])

        # NOTE: training metrics will only be computed on the masked samples, that is, samples that are used for optimization
        rhs = target['rhs'][mask].squeeze()
        rhs_hat = pred['rhs_hat'][mask].squeeze()
        self.train_metrics(rhs_hat, rhs)
        for name, loss in losses.items():
            self.log(f'train.{name}', loss, on_epoch=True, on_step=False, sync_dist=True)
        return losses


    # def on_validation_epoch_start(self):
    #     if self.current_epoch > 3:
    #         # Freeze quantizer parameters
    #         self.apply(torch.ao.quantization.disable_observer)
    #     if self.current_epoch > 2:
    #         # Freeze batch norm mean and variance estimates
    #         self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
    #     # self.quantized_model = torch.ao.quantization.convert(self.to('cpu').eval())
    #     return super().on_validation_epoch_start()
    

    def validation_step(self, sample, batch_idx):
        x = normalize(sample[0], self.s2_mean, self.s2_std)
        rhs = sample[1]
        slope_mask = self.get_slope_mask(sample[3][..., 7, 7])
        veg_mask = get_veg_mask(sample[2])
        # NOTE: veg mask will filter out nonveg samples if filter_out_nonveg is True
        mask = self.add_veg_mask(veg_mask, slope_mask)
        # NOTE: this step will zero out nonveg samples if zero_out_nonveg is True
        rhs = self.process_nonveg(rhs, veg_mask)
        # NOTE: this step will add slope as input if feed_slope is True, sample[3] is the slope array
        x = self.process_slope(x, sample[3])
        # NOTE: this step will add lat and lon as input if feed_latlon is True, sample[4] and sample[5] are the lat and lon vectors
        x = self.process_latlon(x, sample[4], sample[5])
        y_hat = self.forward(x.float())

        losses, pred, target = self.loss_fc(y_hat, mask, rhs, *sample[2:])
        rhs = target['rhs'].squeeze()
        rhs_hat = pred['rhs_hat'].squeeze()
        # Process based on slope evaluation strategy, rhs_hat and rhs are masked if not evaluated on steep slopes

        veg_mask = veg_mask.bool()
        self.val_metrics(rhs_hat[slope_mask], rhs[slope_mask])
        veg_slope_mask = (slope_mask & veg_mask).bool()
        self.val_metrics_veg(rhs_hat[veg_slope_mask], rhs[veg_slope_mask])
        if self.evaluate_high_slope:
            self.val_me_with_steep_slope(rhs_hat, rhs)

        if pred.get('lc_hat') is not None:
            # self.val_metrics_lcc(pred.get('lc_hat'), target.get('lc'))
            self.log(f'{self.val_dataname}.acc', self.val_metrics_lcc, on_epoch=True, on_step=False, sync_dist=True)

        for name, loss in losses.items():
            self.log(f'{self.val_dataname}.{name}', loss, on_epoch=True, on_step=False, sync_dist=True)

        return {
            'veg_mask': veg_mask,
            'slope_mask': slope_mask,
            **pred, **target,
            'rhs_hat': rhs_hat,
            'rhs': rhs,
            'lat': sample[5][:, 7],
            'lon': sample[4][:, 7],
        }
        
    def on_validation_epoch_end(self):
        if hasattr(self, 'train_metrics'): 
            self.log_dict(self.train_metrics.compute(), on_epoch=True, on_step=False, sync_dist=True)

        self.log_dict(self.val_metrics.compute(), on_epoch=True, on_step=False, sync_dist=True)
        self.log_dict(self.val_metrics_veg.compute(), on_epoch=True, on_step=False, sync_dist=True)

    # def on_test_epoch_start(self):
    #     self = self.half()

    def test_step(self, sample, batch_idx):
        '''
        For sparse evaluation
        '''
        x = normalize(sample[0], self.s2_mean, self.s2_std)
        slope_mask = self.get_slope_mask(sample[3][..., 7, 7])
        veg_mask = get_veg_mask(sample[2].long())
        if self.feed_latlon:
            lon = sample[4].unsqueeze(1).repeat(1, 15, 1)
            lat = sample[5].unsqueeze(2).repeat(1, 1, 15)
            sin_lon = torch.sin(lon*torch.pi/180)
            cos_lon = torch.cos(lon*torch.pi/180)
            lat = (lat - self.lat_mean) / self.lat_std
            sin_lon = (sin_lon - self.lon_sin_mean) / self.lon_sin_std
            cos_lon = (cos_lon - self.lon_cos_mean) / self.lon_cos_std
            x = torch.cat([x, lat.unsqueeze(1), sin_lon.unsqueeze(1), cos_lon.unsqueeze(1)], dim=1)
        y_hat = self.forward(x.float())
        y_hat = y_hat[:, :303, 7, 7]#.reshape(-1, 101, 3)
        return y_hat, sample[1], slope_mask.unsqueeze(1), veg_mask.unsqueeze(1)
    
    def on_predict_epoch_start(self):
        # self = self.half()
        self = torch.compile(self)
        if hasattr(self.trainer.datamodule.pred_dataset, 'tile_id'):
            if not self.trainer.datamodule.cache_predictions:
                self.trainer.datamodule.pred_dataset.initialize_output()
            # self.trainer.datamodule.pred_dataset.set_prediction_fname(self.logger._experiment.id)
            self.predict_step = self._predict_step_for_large_tile
        else:            
            self.predict_step = self._predict_step_for_small_patch
            self.trainer.datamodule.pred_dataset.init_out_h5(self.logger._experiment.id)
    
    @torch.no_grad()
    def _predict_step_for_large_tile(self, sample, batch_idx):
        # y_topleft, x_topleft = self.trainer.datamodule.pred_dataset.patch_coords_dict[batch_idx][1]
        if self.feed_latlon:
            x = normalize(sample[0].float(), self.s2_mean, self.s2_std)
            x = torch.cat([x, sample[-1]], dim=1)
        else:
            x = normalize(sample[0].float(), self.s2_mean, self.s2_std)
        # y_hat = self.forward(x.half())
        y_hat = self.forward(x.float())
        self.trainer.datamodule.pred_dataset.write_patch_predictions(y_hat, sample[1], batch_idx)

        
    @torch.no_grad()
    def _predict_step_for_small_patch(self, sample, batch_idx):
        # NOTE: for downstream task
        x = normalize(sample[0].float(), self.s2_mean, self.s2_std)
        x = self.process_latlon(x, sample[3], sample[4])
        if x.isnan().any():
            raise ValueError('x contains nan')
        y_hat = self.forward(x.float())
        self.trainer.datamodule.pred_dataset.write_patch_predictions(y_hat, sample[0], sample[1], sample[5], sample[6], batch_idx)
    
        
    @torch.no_grad()
    def on_predict_end(self):
        if hasattr(self.trainer.datamodule.pred_dataset, 'tile_id'):
            if self.trainer.datamodule.cache_predictions:
                self.trainer.datamodule.pred_dataset.save_predictions()