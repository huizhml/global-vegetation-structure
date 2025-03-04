import torch
import torch.nn as nn
from typing import Any, Dict
from lightning import LightningModule
import torchmetrics
from models.metrics import MAE, RMSE, MAPE, ME

from models.modules.util import get_nonveg_mask
from const import LAT_MEAN, LAT_STD, LON_SIN_MEAN, LON_SIN_STD, LON_COS_MEAN, LON_COS_STD, SLOPE_MEAN, SLOPE_STD



class BaseModel(LightningModule):

    def __init__(
            self, 
            feed_slope: bool = False,
            zero_slope: bool = False,
            slope_th: float = 20, 
            zero_out_nonveg:bool=False,
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
                **{ f'MAE_h>{i}/': MAE(threshold=i) for i in [30,40, 50]},
                **{ f'RMSE_h>{i}/': RMSE(threshold=i) for i in [30,40, 50]},
                **{ f'ME_h>{i}/': ME(threshold=i) for i in [30,40, 50]}
                # # 'mape': MAPE()
            }, compute_groups=False, # update process is different for these metrics
            postfix='--- train'
        )
        self.val_dataname = 'val'
        self.val_metrics = self.train_metrics.clone(postfix=f'--- {self.val_dataname}')
        self.val_metrics_veg = self.train_metrics.clone(postfix=f'--- {self.val_dataname} --- veg')
        self.val_metrics_lcc = torchmetrics.Accuracy('multiclass', num_classes=12, multidim_average='global')
        if self.evaluate_high_slope: # for bias correction
            self.val_me_with_steep_slope = ME()
        self.save_hyperparameters()

    # def on_validation_start(self):
    #     if self.trainer.world_size == 1 and self.current_epoch == 0: # only for validate, test and predictions
    #         # self.val_dataname = f'--- {self.trainer.datamodule.val_fp.stem}'
    #         self.val_metrics.postfix = f'--- {self.val_dataname}'
    #         self.val_metrics_veg.postfix = f'--- {self.val_dataname} --- veg'

   

    # def configure_model(self):
    #     if self.model is not None:
    #         return

    #     for module in self.modules():
    #         if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
    #             fully_shard(module, mesh=self.device_mesh)

    #     fully_shard(model, mesh=self.device_mesh)

    #     self.model = torch.compile(self.encoder)

    def get_slope_mask(self, slope):
        loss_mask_slope = torch.where(slope < self.slope_th, 1, 0)
        return loss_mask_slope.type(torch.bool)

    
    def training_step(self, sample, batch_idx):
        x = self.transform(sample[0])
        rhs = sample[1]
        # print(rhs.mean())
        slope_mask = self.get_slope_mask(sample[3][..., 7, 7])
        if self.filter_out_nonveg:
            veg_mask = get_nonveg_mask(sample[2])
            slope_mask = (veg_mask & slope_mask).bool()
        if self.zero_out_nonveg:
            label_mask = get_nonveg_mask(sample[2])
            rhs = rhs * label_mask.unsqueeze(-1)
        if self.feed_slope:
            slope = torch.nan_to_num(sample[3], nan=0) 
            slope = (slope - SLOPE_MEAN) / SLOPE_STD
            x = torch.cat([x, slope.unsqueeze(1)], dim=1)
        if self.feed_latlon:
            lon = sample[4].unsqueeze(1).repeat(1, 15, 1)
            lat = sample[5].unsqueeze(2).repeat(1, 1, 15)
            sin_lon = torch.sin(lon*torch.pi/180)
            cos_lon = torch.cos(lon*torch.pi/180)
            lat = (lat - LAT_MEAN) / LAT_STD
            sin_lon = (sin_lon - LON_SIN_MEAN) / LON_SIN_STD
            cos_lon = (cos_lon - LON_COS_MEAN) / LON_COS_STD
            x = torch.cat([x,lat.unsqueeze(1), sin_lon.unsqueeze(1), cos_lon.unsqueeze(1)], dim=1)
        y_hat = self.forward(x.float()) # (n, 303, 15, 15)

        losses, pred, target = self.loss_fc(y_hat, slope_mask, rhs, *sample[2:])

        rhs = target['rhs'][slope_mask].squeeze()
        rhs_hat = pred['rhs_hat'][slope_mask].squeeze()
        self.train_metrics(rhs_hat, rhs)
        for name, loss in losses.items():
            self.log(f'train.{name}', loss, on_epoch=True, on_step=False, sync_dist=True)
        return losses

    # def on_before_optimizer_step(self, optimizer):
    #     sch = self.lr_schedulers()
    #     opt = self.optimizers()
    #     import ipdb; ipdb.set_trace()
    #     print(f'lr: {opt.param_groups[0]["lr"]}')
    #     print(sch.get_lr())



    # def lr_scheduler_step(self, scheduler, metric):
    #     re = super().lr_scheduler_step(scheduler, metric)
    #     sch = self.lr_schedulers()
    #     opt = self.optimizers()
    #     import ipdb; ipdb.set_trace()
    #     print(f'lr: {opt.param_groups[0]["lr"]}')
    #     print(sch.get_lr())
    #     return re


    # def on_train_epoch_end(self):
    #     self.log_dict(self.train_metrics.compute(), on_epoch=True, on_step=False, sync_dist=True)
    #     self.trainer.train_dataloader.close()
    #     return super().on_train_epoch_end()

    def on_train_epoch_start(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.val_metrics_veg.reset()
        self.val_metrics_lcc.reset()
        if self.evaluate_high_slope:
            self.val_me_with_steep_slope.reset()
        return super().on_train_epoch_start()

    def on_validation_epoch_end(self):
        if hasattr(self, 'train_metrics'): #TODO
            self.log_dict(self.train_metrics.compute(), on_epoch=True, on_step=False, sync_dist=True)
            # self.trainer.train_dataloader.close()

        self.log_dict(self.val_metrics.compute(), on_epoch=True, on_step=False, sync_dist=True)
        self.log_dict(self.val_metrics_veg.compute(), on_epoch=True, on_step=False, sync_dist=True)
        # self.trainer.val_dataloaders.close()


    def validation_step(self, sample, batch_idx):
        
        x = self.transform(sample[0])
        rhs = sample[1]
        slope_mask = self.get_slope_mask(sample[3][..., 7, 7])
        veg_mask = get_nonveg_mask(sample[2])
        if self.filter_out_nonveg:
            slope_mask = (veg_mask & slope_mask).bool()
        if self.feed_slope:
            if self.zero_slope:
                slope = torch.zeros_like(sample[3])
            else:
                slope = torch.nan_to_num(sample[3], nan=0) 
                slope = (slope - SLOPE_MEAN) / SLOPE_STD
            x = torch.cat([x, slope.unsqueeze(1)], dim=1)
        if self.feed_latlon:
            lon = sample[4].unsqueeze(1).repeat(1, 15, 1)
            lat = sample[5].unsqueeze(2).repeat(1, 1, 15)
            sin_lon = torch.sin(lon*torch.pi/180)
            cos_lon = torch.cos(lon*torch.pi/180)
            lat = (lat - LAT_MEAN) / LAT_STD
            sin_lon = (sin_lon - LON_SIN_MEAN) / LON_SIN_STD
            cos_lon = (cos_lon - LON_COS_MEAN) / LON_COS_STD
            x = torch.cat([x,lat.unsqueeze(1), sin_lon.unsqueeze(1), cos_lon.unsqueeze(1)], dim=1)
        y_hat = self.forward(x.float())
        
        if self.zero_out_nonveg:
            label_mask = get_nonveg_mask(sample[2])
            rhs = rhs * label_mask.unsqueeze(-1)
        losses, pred, target = self.loss_fc(y_hat, slope_mask, rhs, *sample[2:])
        if self.evaluate_high_slope:
            rhs = target['rhs'].squeeze()
            rhs_hat = pred['rhs_hat'].squeeze()
            lc = sample[2]
            self.val_me_with_steep_slope(rhs_hat, rhs)
        else:
            rhs = target['rhs'][slope_mask].squeeze()
            rhs_hat = pred['rhs_hat'][slope_mask].squeeze()
            lc = sample[2][slope_mask]


        if self.filter_out_nonveg:
            self.val_metrics_veg(rhs_hat, rhs)
        else:
            veg_mask = get_nonveg_mask(lc)
            veg_mask = veg_mask.bool()
            self.val_metrics(rhs_hat, rhs)
            self.val_metrics_veg(rhs_hat[veg_mask], rhs[veg_mask])

        if pred.get('lc_hat') is not None:
            self.val_metrics_lcc(pred.get('lc_hat'), target.get('lc'))
            self.log(f'{self.val_dataname}.acc', self.val_metrics_lcc, on_epoch=True, on_step=False, sync_dist=True)
        
        for name, loss in losses.items():
            self.log(f'{self.val_dataname}.{name}', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        return {
            'veg_mask': veg_mask,
            'slope_mask': slope_mask,
            **pred, **target,
            'rhs_hat': rhs_hat,
            'rhs': rhs,  
        }
    
    def on_predict_epoch_start(self):
        self.trainer.datamodule.pred_dataset.set_prediction_fname(self.logger._experiment.id)
        return super().on_predict_epoch_start()

    def predict_step(self, sample, batch_idx):
        if self.feed_latlon:
            x = self.transform(sample[0][0])
            x = torch.cat([x, sample[0][1]], dim=1)
        else:
            x = self.transform(sample[0])
        y_hat = self.forward(x.float())
        self.trainer.datamodule.pred_dataset.write_patch_predictions(y_hat, batch_idx, self.logger._experiment.id)

    def test_step(self, sample, batch_idx):
        x = self.transform(sample[0])
        slope_mask = self.get_slope_mask(sample[3][..., 7, 7])
        veg_mask = get_nonveg_mask(sample[2])
        if self.feed_latlon:
            lon = sample[4].unsqueeze(1).repeat(1, 15, 1)
            lat = sample[5].unsqueeze(2).repeat(1, 1, 15)
            sin_lon = torch.sin(lon*torch.pi/180)
            cos_lon = torch.cos(lon*torch.pi/180)
            lat = (lat - LAT_MEAN) / LAT_STD
            sin_lon = (sin_lon - LON_SIN_MEAN) / LON_SIN_STD
            cos_lon = (cos_lon - LON_COS_MEAN) / LON_COS_STD
            x = torch.cat([x,lat.unsqueeze(1), sin_lon.unsqueeze(1), cos_lon.unsqueeze(1)], dim=1)
        y_hat = self.forward(x.float())
        # pred = y_hat[:, :303].reshape(-1, 101, 3, 15, 15)
        # pred_rh100 = pred[:, 100, 1]
        # print(batch_idx)
        # print((pred[:, 100, 1] - pred[:, 99, 1]).min())
        # print('RH100 mean prediction: ', pred_rh100.mean())
        # print('RH100 min prediction: ', pred_rh100.min())
        # print('RH100 max prediction: ', pred_rh100.max())
        # print('mean error: ', (pred_rh100[:, 7,7] - sample[1][:, 100]).mean())
        # print('mae error: ', (pred_rh100[:, 7,7] - sample[1][:, 100]).abs().mean())
        # print('rmse error: ', ((pred_rh100[:, 7,7] - sample[1][:, 100])**2).mean().sqrt())
        # print('------------------------------------------')
        # import ipdb; ipdb.set_trace()
        y_hat = y_hat[:, :303, 7, 7].reshape(-1, 101, 3)
        canopy_heights = y_hat[:, [95, 98, 100], 1]
        canopy_heights = torch.cat([canopy_heights, sample[1][:, [95,98,100]], slope_mask.unsqueeze(1), veg_mask.unsqueeze(1)], dim=1)
        return canopy_heights
    

