import torchmetrics
import torch.nn as nn
from lightning import LightningModule
import torch
import wandb
import numpy as np
from pathlib import Path
from kornia.enhance import normalize
from wandb.plot.custom_chart import plot_table

from const import VSM_NODATA
from datasets.h5_dataset import resolve_rh_idxs, resolve_diversity, diversity_indices_torch

class NaturalnessMapping(LightningModule):
    def __init__(self,
                 transform: nn.Module = None,
                 loss_fc: nn.Module = None,
                 backbone: nn.Module = None,
                 mean_std_fp: str = None,
                 rh_idxs='rh98',
                 use_s2: bool = False,
                 diversity=None,
                 **kwargs):
        super(NaturalnessMapping, self).__init__()

        self.loss_fc = loss_fc
        self.transform = transform
        # self.backbone = backbone
        if backbone is not None:
            for name, module in backbone.named_children():
                setattr(self, name, module)
        self.forward = backbone.forward

        # Input composition mirrors NaturalnessDataModule's three knobs:
        # rh_idxs (which RH bands) + use_s2 (concat Sentinel-2) + diversity
        # (which RH-derived indices). Auto-wired from the data config in run.py,
        # so the model can never disagree with the dataset. The dataset serves
        # the full 101 profile; selection / diversity / concat happen here on
        # GPU (see on_after_batch_transfer).
        self.rh_idxs = resolve_rh_idxs(rh_idxs)
        self.use_s2 = use_s2
        self.div_idxs = resolve_diversity(diversity)
        self.n_rh = len(self.rh_idxs)
        self.n_div = len(self.div_idxs)
        # Index tensors as (non-persistent) buffers so Lightning moves them to
        # the batch device; not learned, so kept out of the checkpoint.
        self.register_buffer('rh_idx_t', torch.as_tensor(self.rh_idxs, dtype=torch.long), persistent=False)
        self.register_buffer('div_idx_t', torch.as_tensor(self.div_idxs, dtype=torch.long), persistent=False)

        # Channel order is S2 -> RH -> diversity, applied identically here and
        # in on_after_batch_transfer: [s2(12)..., rh(n_rh)..., div(n_div)...].
        mean_std_fp = Path(mean_std_fp).expanduser()
        if not mean_std_fp.exists():
            raise ValueError(f'Mean and std file {mean_std_fp} does not exist')
        file = np.load(mean_std_fp)
        mean_parts, std_parts = [], []
        if self.use_s2:
            mean_parts.append(np.asarray(file['mean_s2']))
            std_parts.append(np.asarray(file['std_s2']))
        if self.n_rh > 0:
            mean_parts.append(np.asarray(file['mean'])[self.rh_idxs])
            std_parts.append(np.asarray(file['std'])[self.rh_idxs])
        if self.n_div > 0:
            if 'mean_div' not in file.files:
                raise ValueError(
                    f'{mean_std_fp} has no diversity stats (mean_div/std_div). '
                    'Regenerate with step6_calculate_naturalness_stats (it now '
                    'also computes diversity-index stats).')
            mean_parts.append(np.asarray(file['mean_div'])[self.div_idxs])
            std_parts.append(np.asarray(file['std_div'])[self.div_idxs])
        if not mean_parts:
            raise ValueError('No input channels: rh_idxs empty, use_s2 False, diversity None')
        self.mean = np.concatenate(mean_parts, axis=0)
        self.std = np.concatenate(std_parts, axis=0)

        # Overall metrics
        self.val_metrics_veg = None
        self.val_metrics_lcc = None
        self.train_metrics = torchmetrics.MetricCollection({
            'acc': torchmetrics.Accuracy('multiclass', num_classes=7, average='micro'),
            'acc_per_class': torchmetrics.Accuracy('multiclass', num_classes=7, average='none'),
            'precision': torchmetrics.Precision('multiclass', num_classes=7, average='micro'),
            'precision_per_class': torchmetrics.Precision('multiclass', num_classes=7, average='none'),
            'recall': torchmetrics.Recall('multiclass', num_classes=7, average='micro'),
            'recall_per_class': torchmetrics.Recall('multiclass', num_classes=7, average='none'),
            'confusion_matrix': torchmetrics.ConfusionMatrix('multiclass', num_classes=7)
        }, compute_groups=False, postfix='-train')
        
        self.val_metrics = self.train_metrics.clone(postfix='-val')

        # Define the class labels
        self.class_labels = ['No forest', 'Natural forest', 'Natural forest (secondary)', 
                             'Planted forest', 'Short rotation plantation', 'Oil palm plantation', 'Agroforestry']

    def normalize(self, x):
        return normalize(x, self.mean, self.std)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        '''Build + normalise the model input on-GPU from the full profile the
        dataset serves: select RH bands, derive diversity, concat in the SAME
        order as self.mean/std -> [s2..., rh..., div...], normalise, then zero
        the RH/diversity channels at nodata pixels (Plan C).

        RH nodata is whole-pixel (all 101 bands share it). Zeroing AFTER
        normalisation makes a nodata pixel exactly 0 == the per-channel mean
        (mean/std were computed excluding nodata), so conv / BatchNorm never
        see the raw VSM_NODATA outlier. No explicit mask channel: the net
        infers "no data" from the all-zero RH.

        S2 is NOT zeroed: it has its own validity (still observed over water /
        built-up where RH is masked), so the RH mask must not touch it.
        '''
        vsm, s2, y, *extra = batch  # extra == [rowid]; passed through untouched
        # whole-pixel RH validity (any band carries it; use band 0) -> (B,1,H,W)
        valid = vsm[:, :1] != VSM_NODATA
        parts = []
        if self.use_s2:
            parts.append(s2.float())
        if self.n_rh > 0:
            parts.append(vsm[:, self.rh_idx_t].float())
        if self.n_div > 0:
            div = diversity_indices_torch(vsm)[:, self.div_idx_t]
            parts.append(torch.nan_to_num(div, nan=0.0))
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        x = self.normalize(x.float())
        # Zero ONLY the RH(+diversity) channels at nodata pixels (in-place slice
        # assignment, no cat); the S2 block (first 12 channels iff use_s2) is
        # left untouched. torch.where broadcasts valid (B,1,H,W) and 0.0.
        s2_c = 12 if self.use_s2 else 0
        x[:, s2_c:] = torch.where(valid, x[:, s2_c:], 0.0)
        # (x, y) for train/val; (x, y, rowid) for test (extra passed through).
        return (x, y, *extra)


    # def on_load_checkpoint(self, checkpoint):
    #     sd = checkpoint.get('state_dict', {})
    #     def add_prefix(k):
    #         if k.startswith('backbone.'):
    #             return k
    #         return f'backbone.{k}'
    #     checkpoint['state_dict'] = {add_prefix(k): v for k, v in sd.items()}
    
    def on_train_epoch_start(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
    
    def training_step(self, batch, batch_idx):
        x, y, *_ = batch  # composed + normalised by on_after_batch_transfer; rowid unused
        y_hat = self(x)
        loss = self.loss_fc(y_hat[:, :, 7,7], y)
        self.log('train.loss', loss)
        self.train_metrics(y_hat[:, :, 7,7], y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch  # composed + normalised by on_after_batch_transfer; rowid unused
        y_hat = self(x)
        loss = self.loss_fc(y_hat[:, :, 7,7], y)
        self.log('val.loss', loss)
        self.val_metrics(y_hat[:, :, 7,7], y)
        return loss

    def test_step(self, batch, batch_idx):
        # rowid is the join key back to reference_data_set (lat/lon, biome...).
        # Returned tuple becomes `outputs` in the prediction-logger callback's
        # on_test_batch_end (same contract as callbacks.prediction_logger).
        x, y, rowid = batch  # composed + normalised by on_after_batch_transfer
        logits = self(x)[:, :, 7, 7]            # center pixel, (B, n_classes)
        probs = torch.softmax(logits, dim=1)
        return probs, y, rowid
    
    def on_validation_epoch_end(self):
        val_metrics = self.val_metrics.compute()
        train_metrics = self.train_metrics.compute()

        # Log overall metrics
        self.log_dict({k.replace('-', '/'): v for k, v in val_metrics.items() if 'per_class' not in k and k != 'confusion_matrix-val'}, 
                      on_epoch=True, on_step=False, sync_dist=True)
        self.log_dict({k.replace('-', '/'): v for k, v in train_metrics.items() if 'per_class' not in k and k != 'confusion_matrix-train'}, 
                      on_epoch=True, on_step=False, sync_dist=True)

        # Log per-class metrics with actual labels
        for metric in ['acc', 'precision', 'recall']:
            for i, label in enumerate(self.class_labels):
                self.log(f'{metric}/per_class/{label}.val', val_metrics[f'{metric}_per_class-val'][i], 
                         on_epoch=True, on_step=False, sync_dist=True)
                self.log(f'{metric}/per_class/{label}.train', train_metrics[f'{metric}_per_class-train'][i], 
                         on_epoch=True, on_step=False, sync_dist=True)
        
        # Log confusion matrices
        # Convert confusion matrix to list of [actual, predicted, count]
        for split, metrics in [('val', val_metrics), ('train', train_metrics)]:
            confusion_data = []
            conf_matrix = metrics[f'confusion_matrix-{split}']
            for i in range(len(self.class_labels)):
                for j in range(len(self.class_labels)):
                    confusion_data.append([
                        self.class_labels[i],  # Actual class
                        self.class_labels[j],  # Predicted class 
                        conf_matrix[i,j].item() # Number of samples
                    ])

            table = plot_table(
                data_table=wandb.Table(
                    columns=["Actual", "Predicted", "nPredictions"],
                    data=confusion_data
                ),
                vega_spec_name="wandb/confusion_matrix/v1", 
                fields={
                    "Actual": "Actual",
                    "Predicted": "Predicted", 
                    "nPredictions": "nPredictions"
                },
                string_fields={"title": f'confusion_matrix/{split}'},
                split_table=False
            )
            self.logger.experiment.log({f'confusion_matrix/{split}': table})
