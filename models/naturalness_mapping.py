import torchmetrics
import torch.nn as nn
from lightning import LightningModule
import torch
import wandb
import numpy as np
from pathlib import Path
from kornia.enhance import normalize
from wandb.plot.custom_chart import plot_table

class NaturalnessMapping(LightningModule):
    def __init__(self, 
                 transform: nn.Module = None,
                 loss_fc: nn.Module = None,
                 backbone: nn.Module = None,
                 mean_std_fp: str = None,
                 **kwargs):
        super(NaturalnessMapping, self).__init__()
        
        self.loss_fc = loss_fc
        self.transform = transform
        # self.backbone = backbone
        if backbone is not None:
            for name, module in backbone.named_children():
                setattr(self, name, module)
        self.forward = backbone.forward
        
        # Get mean and std of the input data
        mean_std_fp = Path(mean_std_fp).expanduser()
        if not mean_std_fp.exists():
            raise ValueError(f'Mean and std file {mean_std_fp} does not exist')
        file = np.load(mean_std_fp)
        if backbone.in_channels == 1: # rh98 only
            self.mean = file['mean'][98:99]
            self.std = file['std'][98:99]
        if backbone.in_channels == 12: # s2 only
            self.mean = file['mean_s2']
            self.std = file['std_s2']
        elif backbone.in_channels == 13: # s2 and rh98
            mean_top_height = file['mean'][98:99]
            std_top_height = file['std'][98:99]
            mean_s2 = file['mean_s2']
            std_s2 = file['std_s2']
            self.mean = np.concatenate([mean_s2, mean_top_height], axis=0)
            self.std = np.concatenate([std_s2, std_top_height], axis=0)
        elif backbone.in_channels == 101: # rhs only
            self.mean = file['mean']
            self.std = file['std']
        elif backbone.in_channels == 113: # s2 and rhs
            mean_top_height = file['mean'][98:99]
            std_top_height = file['std'][98:99]
            mean_s2 = file['mean_s2']
            std_s2 = file['std_s2']
            self.mean = np.concatenate([mean_s2, mean_top_height], axis=0)
            self.std = np.concatenate([std_s2, std_top_height], axis=0)

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
        rhs, s2, y = batch
        if self.mean.shape[0] == 1 or self.mean.shape[0] == 101:
            x = rhs
        elif self.mean.shape[0] == 12:
            x = s2
        elif self.mean.shape[0] == 13 or self.mean.shape[0] == 113:
            x = torch.cat([rhs, s2], dim=1)
        x = self.normalize(x.float())
        y_hat = self(x)
        loss = self.loss_fc(y_hat[:, :, 7,7], y)
        self.log('train.loss', loss)
        self.train_metrics(y_hat[:, :, 7,7], y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        rhs, s2, y = batch
        if self.mean.shape[0] == 1 or self.mean.shape[0] == 101:
            x = rhs
        elif self.mean.shape[0] == 12:
            x = s2
        elif self.mean.shape[0] == 13 or self.mean.shape[0] == 113:
            x = torch.cat([rhs, s2], dim=1)
        y_hat = self(x.float())
        loss = self.loss_fc(y_hat[:, :, 7,7], y)
        self.log('val.loss', loss)
        self.val_metrics(y_hat[:, :, 7,7], y)
        return loss
    
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
