import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
import torchmetrics
import numpy as np
from lightning import Trainer, LightningModule
from modelopt.torch.quantization.utils import export_torch_mode
from models.metrics import MAE, RMSE, MAPE, ME
from utils import print_size_of_model
from models.modules.util import get_veg_mask
from const import LAT_MEAN, LAT_STD, LON_SIN_MEAN, LON_SIN_STD, LON_COS_MEAN, LON_COS_STD, SLOPE_MEAN, SLOPE_STD

MASKED_VALUE = 32767

class BaseModel(LightningModule):

    def __init__(
            self,
            feed_slope: bool = False,
            zero_slope: bool = False,
            slope_th: float = 20,
            zero_out_nonveg: bool = False,
            filter_out_nonveg: bool = False,
            feed_latlon: bool = False,
            evaluate_high_slope: bool = False,
            loss_fc: nn.Module = None,
            transform: nn.Module = None,
            yhat_transform: nn.Module = None, *args: Any, **
            kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
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
        slope = (slope - SLOPE_MEAN) / SLOPE_STD
        return torch.cat([x, slope.unsqueeze(1)], dim=1)

    def _process_latlon(self, x, lon, lat):
        lon = lon.unsqueeze(1).repeat(1, 15, 1)
        lat = lat.unsqueeze(2).repeat(1, 1, 15)
        sin_lon = torch.sin(lon * torch.pi / 180)
        cos_lon = torch.cos(lon * torch.pi / 180)
        lat = (lat - LAT_MEAN) / LAT_STD
        sin_lon = (sin_lon - LON_SIN_MEAN) / LON_SIN_STD
        cos_lon = (cos_lon - LON_COS_MEAN) / LON_COS_STD
        return torch.cat([x, lat.unsqueeze(1), sin_lon.unsqueeze(1), cos_lon.unsqueeze(1)], dim=1)

    def _process_nonveg(self, rhs, veg_mask):
        return rhs * veg_mask.unsqueeze(-1)

    def _add_veg_mask(self, veg_mask, slope_mask):
        return (veg_mask & slope_mask).bool()

    
    def on_train_epoch_start(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.val_metrics_veg.reset()
        self.val_metrics_lcc.reset()
        if self.evaluate_high_slope:
            self.val_me_with_steep_slope.reset()
        return super().on_train_epoch_start()
    
    
    def training_step(self, sample, batch_idx):
        x = self.transform(sample[0])
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
        
    #     # Create quantized model without deepcopy
    #     self.quantized_model = torch.ao.quantization.convert(self.to('cpu').eval())
        return super().on_validation_epoch_start()
    

    def validation_step(self, sample, batch_idx):
        x = self.transform(sample[0])
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
        # y_hat = self.quantized_model.forward(x)

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
        
    # def on_validation_epoch_end(self):
    #     if hasattr(self, 'train_metrics'):  # TODO
    #         self.log_dict(self.train_metrics.compute(), on_epoch=True, on_step=False, sync_dist=True)

    #     self.log_dict(self.val_metrics.compute(), on_epoch=True, on_step=False, sync_dist=True)
    #     self.log_dict(self.val_metrics_veg.compute(), on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, sample, batch_idx):
        '''
        For sparse evaluation
        '''
        x = self.transform(sample[0])
        slope_mask = self.get_slope_mask(sample[3][..., 7, 7])
        veg_mask = get_veg_mask(sample[2])
        if self.feed_latlon:
            lon = sample[4].unsqueeze(1).repeat(1, 15, 1)
            lat = sample[5].unsqueeze(2).repeat(1, 1, 15)
            sin_lon = torch.sin(lon*torch.pi/180)
            cos_lon = torch.cos(lon*torch.pi/180)
            lat = (lat - LAT_MEAN) / LAT_STD
            sin_lon = (sin_lon - LON_SIN_MEAN) / LON_SIN_STD
            cos_lon = (cos_lon - LON_COS_MEAN) / LON_COS_STD
            x = torch.cat([x, lat.unsqueeze(1), sin_lon.unsqueeze(1), cos_lon.unsqueeze(1)], dim=1)
        y_hat = self.forward(x.float())
        y_hat = y_hat[:, :303, 7, 7].reshape(-1, 101, 3)
        return y_hat[:, :, 1], sample[1], slope_mask.unsqueeze(1), veg_mask.unsqueeze(1)


    def _apply_masks(self, prediction, scl, x_topleft, y_topleft):
        prediction_no_border = prediction[:, self.rh_idx,
                                          self.border:self.patch_size - self.border,
                                          self.border:self.patch_size - self.border
                                          ]
        
        
        location_key = f'{y_topleft}_{x_topleft}'
        if location_key not in self.prediction_cache:
            self.prediction_cache[location_key] = prediction_no_border
            return None
        else: # ready to write
            prediction_no_border = torch.cat([self.prediction_cache[location_key], prediction_no_border], dim=0)
            
            # Prepare masks before applying them to reduce repeated operations
            masks_to_apply = []
            
            # if self.mask_empty:
            #     # pixels where all RGB values equal zero are empty (bands B02, B03, B04)
            #     # note self.image has shape: (height, width, channels)
            #     img = self.store[f'{self.tile_id}/s2'][:, 1:4, y_topleft:y_topleft + self.patch_size_no_border, x_topleft:x_topleft + self.patch_size_no_border]
            #     # Use torch operations directly instead of numpy sum
            #     img_tensor = torch.from_numpy(img).to(prediction_no_border.device)
            #     invalid_mask = torch.sum(img_tensor, dim=1, keepdim=True) == 0
            #     masks_to_apply.append(invalid_mask)
            
            if self.mask_with_scl:
                # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
                if scl.shape[2] == self.patch_size:
                    scl = scl[:, :, self.border:self.patch_size - self.border,
                            self.border:self.patch_size - self.border]
                # Convert to torch tensor once and use isin equivalent with pre-computed tensor
                if self.scl_exclude_labels_tensor is None or self.scl_exclude_labels_tensor.device != prediction_no_border.device:
                    self.scl_exclude_labels_tensor = torch.tensor(self.scl_exclude_labels, device=prediction_no_border.device)
                scl_mask = torch.isin(scl, self.scl_exclude_labels_tensor)
                masks_to_apply.append(scl_mask)
            
            # Apply all masks at once using logical_or to combine them
            if len(masks_to_apply) > 1:
                combined_mask = torch.logical_or(*masks_to_apply)
            else:
                combined_mask = masks_to_apply[0]
            prediction_no_border = torch.where(combined_mask, torch.nan, prediction_no_border)
            
            # aggregate predictions with median
            prediction_no_border, _ = torch.nanmedian(prediction_no_border, dim=0)          
            prediction_no_border.round_()  # .mul_(10) In-place operations
            prediction_no_border = torch.nan_to_num(prediction_no_border, nan=MASKED_VALUE).to(torch.int16)
            
            # Move to CPU once and do all numpy operations together
            prediction_no_border = prediction_no_border.cpu().numpy()
            return prediction_no_border
    
    
    
    # def on_predict_epoch_start(self):
    #     if hasattr(self.trainer.datamodule.pred_dataset, 'tile_id'):
    #         self.trainer.datamodule.pred_dataset.set_prediction_fname(self.logger._experiment.id)
    #         self.predict_step = self._predict_step_for_large_tile
    #         self.on_predict_epoch_end = self.trainer.datamodule.pred_dataset.finalize_output
    #     else:            
    #         self.predict_step = self._predict_step_for_small_patch
    #         self.trainer.datamodule.pred_dataset.init_out_h5(self.logger._experiment.id)
    #     return super().on_predict_epoch_start()
    
    # def on_predict_epoch_end(self):
    #     if hasattr(self.trainer.datamodule.pred_dataset, 'zarr_output'):
    #         t0 = time.time()
    #         self.trainer.datamodule.pred_dataset.zarr_output.close()
    #         t1 = time.time()
    #         print(f'Time taken to close zarr output: {t1 - t0} seconds')
    #     return super().on_predict_epoch_end()
    def predict_step(self, sample, batch_idx): #TODO
        return self._predict_step_for_large_tile(sample, batch_idx)



    def _predict_step_for_large_tile(self, sample, batch_idx):
        y_topleft, x_topleft = self.patch_coords_dict[batch_idx][1:]
        y_topleft = y_topleft + self.border
        x_topleft = x_topleft + self.border 
        if self.feed_latlon:
            x = self.transform(sample[0])
            x = torch.cat([x, sample[-1]], dim=1)
        else:
            x = self.transform(sample[0])
        w, h = x.shape[2:]
        pad_w = self.trainer.datamodule.pred_dataset.patch_size - w
        pad_h = self.trainer.datamodule.pred_dataset.patch_size - h
        if y_topleft == 0:
            pad_h = (self.border, 0)
        elif y_topleft + self.patch_size > self.img_height:
            pad_h = (0, self.border)
        else:
            pad_h = (0, 0)
        if x_topleft == 0:
            pad_w = (self.border, 0)
        elif x_topleft + self.patch_size > self.img_width:
            pad_w = (0, self.border)
        else:
            pad_w = (0, 0)
        x = F.pad(x, (*pad_w, *pad_h), 'reflect')
        y_hat = self.forward(x.float())

   
        prediction_no_border = self._apply_masks(y_hat, sample[1], x_topleft, y_topleft)
        if prediction_no_border is None:
            return
        
        self.full_pred[:,y_topleft:y_topleft + self.patch_size_no_border, x_topleft:x_topleft + self.patch_size_no_border] = prediction_no_border
        del self.prediction_cache[f'{y_topleft}_{x_topleft}']
        

    def _predict_step_for_small_patch(self, sample, batch_idx):
        # NOTE: for downstream task
        x = self.transform(sample[0])
        x = self.process_latlon(x, sample[3], sample[4])
        y_hat = self.forward(x.float())
        self.trainer.datamodule.pred_dataset.write_patch_predictions(y_hat, sample[1], sample[5], sample[6], batch_idx)
    
    
    
    @torch.no_grad()
    def on_predict_start(self) -> None:
        
        self.img_height = self.trainer.datamodule.pred_dataset.img_height
        self.img_width = self.trainer.datamodule.pred_dataset.img_width
        self.border = self.trainer.datamodule.pred_dataset.border
        self.patch_size = self.trainer.datamodule.pred_dataset.patch_size
        self.patch_size_no_border = self.trainer.datamodule.pred_dataset.patch_size_no_border
        self.patch_coords_dict = self.trainer.datamodule.pred_dataset.patch_coords_dict
        self.mask_with_scl = self.trainer.datamodule.pred_dataset.mask_with_scl
        self.rh_idx = self.trainer.datamodule.pred_dataset.rh_idx
        self.prediction_cache = {}
        self.nodata_value = self.trainer.datamodule.pred_dataset.nodata_value
        self.scl_exclude_labels = self.trainer.datamodule.pred_dataset.scl_exclude_labels
        self.scl_exclude_labels_tensor = torch.tensor(self.scl_exclude_labels, device=self.device)
        print(f'img_height: {self.img_height}, img_width: {self.img_width}, border: {self.border}, patch_size_no_border: {self.patch_size_no_border}')
        self.full_pred = np.full((len(self.rh_idx), self.img_height, self.img_width), MASKED_VALUE, dtype=np.int16)
        self.trainer.datamodule.pred_dataset.set_prediction_fname(self.logger._experiment.id)
        
        
    @torch.no_grad()
    def on_predict_end(self):
        self.trainer.datamodule.pred_dataset.save_predictions_as_cog(self.full_pred)