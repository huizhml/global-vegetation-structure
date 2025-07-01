import os.path
from typing import List
from collections import OrderedDict
import torch
import torch.nn as nn
import lightning as L
from torch.hub import download_url_to_file
import torchmetrics
import wandb
from wandb.plot.custom_chart import plot_table
from ._base_pl_model import BaseModel
from models.modules.xception_blocks import PointwiseBlock, DoubleSepConvBlock, conv1x1
from utils import get_class


class ResLayer(nn.Module):
    def __init__(self, in_channels, filters):
        super(ResLayer, self).__init__()
        self.filters = filters
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=1, stride=1, bias=True)
        self.w2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class GeoPriorNet(nn.Module):
    """
    This is a fully convolutional version of the GeoPrior FCN proposed by Mac Aodha et al. (2019)
    """

    def __init__(self, in_channels, filters=256):
        super(GeoPriorNet, self).__init__()

        self.feats = nn.Sequential(conv1x1(in_channels=in_channels, out_channels=filters),
                                   nn.ReLU(inplace=True),
                                   ResLayer(in_channels=filters, filters=filters),
                                   ResLayer(in_channels=filters, filters=filters),
                                   ResLayer(in_channels=filters, filters=filters),
                                   ResLayer(in_channels=filters, filters=filters))

        self.geo_scale = conv1x1(in_channels=filters, out_channels=1)
        self.geo_shift = conv1x1(in_channels=filters, out_channels=1)

    def forward(self, x):
        x = self.feats(x)
        scale = self.geo_scale(x) + 1  # initially around 1
        shift = self.geo_shift(x)      # initially around 0
        return scale, shift


def ELUplus1(x):
    elu = nn.ELU(inplace=False)(x)
    return torch.add(elu, 1.0)


def clamp_exp(x, min_x=-100, max_x=10):
    x = torch.clamp(x, min=min_x, max=max_x)
    return torch.exp(x)


class XceptionS2(BaseModel):
    """ A custom fully convolutional neural network designed for pixel-wise analysis of Sentinel-2 satellite images.

    "XceptionS2" builds on the separable convolution described by Chollet (2017) who proposed the Xception network.
    Any kind of down sampling is avoided (no pooling, striding, etc.).

    This architecture is adapted from:
    Lang, N., Schindler, K., Wegner, J.D.: Country-wide high-resolution vegetation height mapping with Sentinel-2,
    Remote Sensing of Environment, vol. 233 (2019) <https://arxiv.org/abs/1904.13270>

    Here, we extend the model class XceptionS2 with the option to estimate pixel-wise uncertainties in regression tasks
    and include an option to add a long skip connection.
    These options are used in:
    Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2022). A high-resolution canopy height model of the Earth.
    arXiv preprint arXiv:2204.08322.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (Set >1 for multi-task learning)
        num_sepconv_blocks (int): Number of blocks
        num_sepconv_filters (int): Number of filters
        returns (string): Key specifying the return. Choices: ['targets', 'variances_exp', 'variances']
        var_activation (string): Set which activation is applied on output variance. Choices ['relu', 'elu', 'exp']
        min_var (float): Shift the output variance by adding min_var.
        detach_var_input (bool): Detach graph before computing the variance. (obsolete)
        long_skip (bool): Add a long skip (residual) connection from the entry block features to the last features.
        manual_init (bool): Option to use a custom initialization setting.
        freeze_features (bool): Option to freeze feature extractor.
        freeze_last_mean (bool): Option to freeze last linear layer that outputs the mean
        freeze_last_var (bool):  Option to freeze last linear layer that outputs the variance
        geo_shift (bool): Option to shift the prediction by a learned shifting prior given latitude longitude
        geo_scale (bool): Option to scale the prediction by a learned scaling prior given latitude longitude
        separate_lat_lon (bool): Option to learn an image encoder (without latitude longitude) and a separate lat lon encoder.
        model_weights_path (string): Path to load pretrained model weights used to initialize
    """

    def __init__(self, 
                 
                 activation_layer: nn.Module,
                 norm_layer: str,
                 mid_block: str,
                 sepconv_block: str,
                 nonlin_block: str,
                 in_channels:int, 
                 entry_block_filters:List[int]=[64, 128],
                 out_channels:int=1, 
                 num_nonlin_blocks:int=2,
                 num_sepconv_blocks:int=8, 
                 num_sepconv_filters:int=728, 
                 long_skip:bool=False, manual_init:bool=False, restrict_rf:bool=True, mlp_skip:bool=False,**kwargs):

        yhat_transform = lambda x: x
        super(XceptionS2, self).__init__(**kwargs, yhat_transform=yhat_transform)

        self.activation_layer = activation_layer
        self.norm_layer = norm_layer

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_sepconv_blocks = num_sepconv_blocks
        self.num_sepconv_filters = num_sepconv_filters
        self.long_skip = long_skip
        self.mlp_skip = mlp_skip
        self.entry_block = PointwiseBlock(activation_layer, in_channels=in_channels, filters=entry_block_filters, out_channels=num_sepconv_filters, norm_layer=norm_layer)
        self.sepconv_blocks = self._make_sepconv_blocks(block=sepconv_block, num_blocks=num_sepconv_blocks)
        if restrict_rf:
            mid_block = get_class(mid_block)
            self.mid_block = mid_block(self.activation_layer, in_channels=self.num_sepconv_filters, out_channels=self.num_sepconv_filters, norm_layer=self.norm_layer)
            self.nonlin_blocks = self._make_sepconv_blocks(block=nonlin_block, kernerl_sizes=(1, 1), num_blocks=num_nonlin_blocks)
        else:
            self.mid_block = nn.Identity()
            self.nonlin_blocks = nn.Identity()

        self.last_conv = conv1x1(in_channels=num_sepconv_filters, out_channels=out_channels)

        # initialize parameters
        if manual_init:
            self._manual_init()

            # train (unfreeze) the last layer(s) of the linear regressor
            if not self.freeze_last_mean:
                print('Unfreeze last layer (mean regressor)... args.freeze_last_mean={}'.format(self.freeze_last_mean))
                for param in self.last_conv.parameters():
                    param.requires_grad = True


    def forward(self, x):
        """
        Args:
            x: input tensor: first 12 channels are sentinel-2 bands, last 3 channels are lat lon encoding
        """
        x = self.entry_block(x)
        if self.long_skip:
            shortcut = x
        x = self.sepconv_blocks(x)
        x = self.mid_block(x)
        if self.mlp_skip:
            conv_features = x
        x = self.nonlin_blocks(x)
        if self.long_skip:
            x = x + shortcut
        if self.mlp_skip:
            x = x + conv_features
        predictions = self.last_conv(x)

        return predictions
    

    def _make_sepconv_blocks(self,kernerl_sizes=(3, 3), block:str='DoubleSepConvBlock', num_blocks:int=None):
        block = get_class(block)
        blocks = []
        for i in range(num_blocks):
            blocks.append(block(self.activation_layer, self.norm_layer, 
                                       in_channels=self.num_sepconv_filters,
                                       filters=[self.num_sepconv_filters, self.num_sepconv_filters],
                                       kernel_sizes=kernerl_sizes))
        return nn.Sequential(*blocks)

    def _manual_init(self):
        print('Manual weight init...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') TODO: check if kaiming would be better with ReLU (see torchvision resnet)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # gamma
                nn.init.constant_(m.bias, 0)  # beta
    

class XceptionS2MixOrder(BaseModel):
    """ A custom fully convolutional neural network designed for pixel-wise analysis of Sentinel-2 satellite images.

    "XceptionS2" builds on the separable convolution described by Chollet (2017) who proposed the Xception network.
    Any kind of down sampling is avoided (no pooling, striding, etc.).

    This architecture is adapted from:
    Lang, N., Schindler, K., Wegner, J.D.: Country-wide high-resolution vegetation height mapping with Sentinel-2,
    Remote Sensing of Environment, vol. 233 (2019) <https://arxiv.org/abs/1904.13270>

    Here, we extend the model class XceptionS2 with the option to estimate pixel-wise uncertainties in regression tasks
    and include an option to add a long skip connection.
    These options are used in:
    Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2022). A high-resolution canopy height model of the Earth.
    arXiv preprint arXiv:2204.08322.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (Set >1 for multi-task learning)
        num_sepconv_blocks (int): Number of blocks
        num_sepconv_filters (int): Number of filters
        returns (string): Key specifying the return. Choices: ['targets', 'variances_exp', 'variances']
        var_activation (string): Set which activation is applied on output variance. Choices ['relu', 'elu', 'exp']
        min_var (float): Shift the output variance by adding min_var.
        detach_var_input (bool): Detach graph before computing the variance. (obsolete)
        long_skip (bool): Add a long skip (residual) connection from the entry block features to the last features.
        manual_init (bool): Option to use a custom initialization setting.
        freeze_features (bool): Option to freeze feature extractor.
        freeze_last_mean (bool): Option to freeze last linear layer that outputs the mean
        freeze_last_var (bool):  Option to freeze last linear layer that outputs the variance
        geo_shift (bool): Option to shift the prediction by a learned shifting prior given latitude longitude
        geo_scale (bool): Option to scale the prediction by a learned scaling prior given latitude longitude
        separate_lat_lon (bool): Option to learn an image encoder (without latitude longitude) and a separate lat lon encoder.
        model_weights_path (string): Path to load pretrained model weights used to initialize
    """

    def __init__(self, 
                 activation_layer: nn.Module,
                 norm_layer: str,
                 mid_block: str,
                 sepconv_block: str,
                 nonlin_block: str,
                 in_channels:int, 
                 entry_block_filters:List[int]=[64, 128],
                 out_channels:int=1, 
                 num_nonlin_blocks:int=2,
                 num_sepconv_blocks:int=8, 
                 num_sepconv_filters:int=728, 
                 long_skip:bool=False, manual_init:bool=False, nonlinear_order:str='last',**kwargs):

        yhat_transform = lambda x: x
        super(XceptionS2MixOrder, self).__init__(**kwargs, yhat_transform=yhat_transform)

        self.activation_layer = activation_layer
        self.norm_layer = norm_layer

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_sepconv_blocks = num_sepconv_blocks
        self.num_sepconv_filters = num_sepconv_filters
        self.long_skip = long_skip
        self.entry_block = PointwiseBlock(activation_layer, in_channels=in_channels, filters=entry_block_filters, out_channels=num_sepconv_filters, norm_layer=norm_layer)
        sepconv_blocks = self._make_sequential_blocks(block=sepconv_block, num_blocks=num_sepconv_blocks)
        mid_block = get_class(mid_block)
        mid_block = mid_block(self.activation_layer, in_channels=self.num_sepconv_filters, out_channels=self.num_sepconv_filters, norm_layer=self.norm_layer)
        nonlin_blocks = self._make_sequential_blocks(block=nonlin_block, kernerl_sizes=(1, 1), num_blocks=num_nonlin_blocks)

        if nonlinear_order == 'last':
            self.layers = nn.Sequential(OrderedDict([
                ('sepconv_blocks', sepconv_blocks), 
                ('mid_block', mid_block), 
                ('nonlin_blocks', nonlin_blocks)
                ]))
        elif nonlinear_order == 'first':
            self.layers = nn.Sequential(OrderedDict([
                ('nonlin_blocks', nonlin_blocks), 
                ('sepconv_blocks', sepconv_blocks), 
                ('mid_block', mid_block)
                ]))
        elif nonlinear_order == 'inbetween':
            self.layers = nn.Sequential()
            for i, (sepconv_block, nonlin_block) in enumerate(zip(sepconv_blocks, nonlin_blocks)):
                self.layers.add_module(f'nonlin_block{i}', nonlin_block)
                self.layers.add_module(f'sepconv_block{i}', sepconv_block)
            self.layers.add_module(f'nonlin_block{i+1}', nonlin_blocks[-1])
            self.layers.add_module('mid_block', mid_block)
        elif nonlinear_order == 'mixed':
            from models.modules.xception_blocks import SepNonlinConvBlock
            self.layers = nn.Sequential()
            for i in range(7): # 7 3x3 conv layers makes receptive field 15
                self.layers.add_module(f'sepconv_block{i}', 
                                       SepNonlinConvBlock(self.activation_layer, self.norm_layer, in_channels=self.num_sepconv_filters,
                                    filters=[self.num_sepconv_filters, self.num_sepconv_filters],
                                    kernel_sizes=(3, 3)))
        else:
            raise ValueError('nonlinear_order must be either "last" or "first" or "inbetween"')
        self.last_conv = conv1x1(in_channels=num_sepconv_filters, out_channels=out_channels)

        # initialize parameters
        if manual_init:
            self._manual_init()

            # train (unfreeze) the last layer(s) of the linear regressor
            if not self.freeze_last_mean:
                print('Unfreeze last layer (mean regressor)... args.freeze_last_mean={}'.format(self.freeze_last_mean))
                for param in self.last_conv.parameters():
                    param.requires_grad = True
                    

        # self.fuse_model()
        # self.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        # torch.ao.quantization.prepare_qat(self, inplace=True) # performs fake quantization
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()
                    
    
    def fuse_model(self):
        for name, module in self.named_modules():
            if 'ConvNormActivation' in name.split('.')[-1]:
                torch.ao.quantization.fuse_modules_qat(module, ['0', '1', '2'], inplace=True)
            elif 'ConvNorm' in name.split('.')[-1]:
                torch.ao.quantization.fuse_modules_qat(module, ['0', '1'], inplace=True)
            elif name.split('.')[-1] == 'shortcut' and len(module) == 2:
                torch.ao.quantization.fuse_modules_qat(module, ['conv_shortcut', 'bn_shortcut'], inplace=True)


    def forward(self, x):
        """
        Args:
            x: input tensor: first 12 channels are sentinel-2 bands, last 3 channels are lat lon encoding
        """
        # x = self.quant(x)
        x = self.entry_block(x)
        if self.long_skip:
            shortcut = x
        x = self.layers(x)
        if self.long_skip:
            x = x + shortcut
        predictions = self.last_conv(x)
        # predictions = self.dequant(predictions)
        return predictions
    


    def _make_sequential_blocks(self,kernerl_sizes=(3, 3), block:str='DoubleSepConvBlock', num_blocks:int=None):
        block = get_class(block)
        blocks = nn.Sequential()
        for i in range(num_blocks):
            blocks.add_module(f'block{i}', block(self.activation_layer, self.norm_layer,
                                       in_channels=self.num_sepconv_filters,
                                       filters=[self.num_sepconv_filters, self.num_sepconv_filters],
                                       kernel_sizes=kernerl_sizes))
        return blocks

    def _manual_init(self):
        print('Manual weight init...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') TODO: check if kaiming would be better with ReLU (see torchvision resnet)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # gamma
                nn.init.constant_(m.bias, 0)  # beta

    # def _constrain_variances(self, variances_tmp):
    #     variances_tmp = self.var_activation(variances_tmp)
    #     variances = variances_tmp + self.min_var
    #     return variances

    def _load_model_weights(self, model_weights_path):
        checkpoint = torch.load(model_weights_path)
        model_weights = checkpoint['model_state_dict']
        self.load_state_dict(model_weights)


def xceptionS2_18blocks(in_channels=12, out_channels=1):
    """
    The model described in:
    'Country-wide high-resolution vegetation height mapping with Sentinel-2' <https://arxiv.org/abs/1904.13270>

    Args:
        in_channels (int): Number of channels/bands of the multi-spectral input image.
        out_channels (int): Dimension of the pixel-wise output.
    """
    return XceptionS2(in_channels=in_channels, out_channels=out_channels, num_sepconv_blocks=18,
                      num_sepconv_filters=728)


def xceptionS2_08blocks(in_channels=12, out_channels=1):
    """
    A smaller version (with only 8 sepconv blocks) of the model described in:
    'Country-wide high-resolution vegetation height mapping with Sentinel-2' <https://arxiv.org/abs/1904.13270>

    Args:
        in_channels (int): Number of channels/bands of the multi-spectral input image.
        out_channels (int): Dimension of the pixel-wise output.
    """
    return XceptionS2(in_channels=in_channels, out_channels=out_channels, num_sepconv_blocks=8,
                      num_sepconv_filters=728)


def xceptionS2_08blocks_256(in_channels=15, out_channels=1, model_weights=None,
                            returns="variances_exp",
                            download_dir="./trained_models",
                            url_trained_models="https://github.com/langnico/global-canopy-height-model/releases/download/v1.0-trained-model-weights/trained_models_GLOBAL_GEDI_2019_2020.zip"):
    """
    The model used in 'A high-resolution canopy height model of the Earth.'
    It is a smaller version (with only 8 sepconv blocks and 256 sepconv filters) of the model described in:
    'Country-wide high-resolution vegetation height mapping with Sentinel-2' <https://arxiv.org/abs/1904.13270>

    Args:
        in_channels (int): Number of channels of the input. (12 sentinel-2 bands + 3 lat-lon-encoding) = 15 channels)
        out_channels (int): Dimension of the pixel-wise output.
        returns (string): Key specifying the return. Choices: ['targets', 'variances_exp', 'variances']
        model_weights (string): This can either be set to the checkpoint path ".pt" or to one of the options below.

    Model weights choices:
        None: Randomly initialize the model weights.
        Path: Path to a pretrained checkpoint file. (E.g. './trained_models/GLOBAL_GEDI_2019_2020/model_0/FT_Lm_SRCB/checkpoint.pt')
        'GLOBAL_GEDI_MODEL_0': This will download the pretrained models and load the fine-tuned model with id 0
        'GLOBAL_GEDI_MODEL_1': This will download the pretrained models and load the fine-tuned model with id 1.
        'GLOBAL_GEDI_MODEL_2': This will download the pretrained models and load the fine-tuned model with id 2.
        'GLOBAL_GEDI_MODEL_3': This will download the pretrained models and load the fine-tuned model with id 3.
        'GLOBAL_GEDI_MODEL_4': This will download the pretrained models and load the fine-tuned model with id 4.

    """

    # download model weights if folder does not exist
    if model_weights is None:
        pass
    elif "GLOBAL_GEDI_MODEL_" in model_weights:
        print("model_weights set to: ", model_weights)

        zip_path = os.path.join(download_dir, "trained_models_GLOBAL_GEDI_2019_2020.zip")
        model_parent_path = os.path.join(download_dir, "GLOBAL_GEDI_2019_2020")
        # get model id and set pretrained model weights path
        model_id = model_weights.split("_")[-1]
        assert model_id in [str(i) for i in range(5)]
        model_weights = os.path.join(
            download_dir, "GLOBAL_GEDI_2019_2020/model_{}/FT_Lm_SRCB/checkpoint.pt".format(model_id))

        if not os.path.exists(model_parent_path):
            print("downloading pretrained models...")
            os.system("mkdir -p {}".format(download_dir))
            download_url_to_file(url=url_trained_models, dst=zip_path, hash_prefix=None, progress=True)
            print("unzipping...")
            os.system("unzip {} -d {}".format(zip_path, download_dir))
            os.system("rm {}".format(zip_path))
        else:
            print("Skipping download. The directory exists already: ", model_parent_path)

    return XceptionS2(in_channels=in_channels, out_channels=out_channels, num_sepconv_blocks=8,
                      num_sepconv_filters=256, returns=returns,
                      long_skip=True,
                      model_weights_path=model_weights)


class XceptionDownstream(XceptionS2MixOrder):
    def __init__(self, in_channels=12, out_channels=1, 
                 **kwargs):
        super(XceptionDownstream, self).__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.val_metrics_veg = None
        self.val_metrics_lcc = None
        
        # Overall metrics
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
        self.class_labels = ['No forest', 'Natural forest (primary)', 'Natural forest (secondary)', 
                             'Planted forest', 'Short rotation plantation', 'Oil palm plantation', 'Agroforestry']

    def on_train_epoch_start(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform(x)
        y_hat = self(x.float())
        loss = self.loss_fc(y_hat[:, :, 7,7], y)
        self.log('train.loss', loss)
        self.train_metrics(y_hat[:, :, 7,7], y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform(x)
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

if __name__ == "__main__":

    # create the model as used in "A high-resolution canopy height model of the Earth."
    model = XceptionS2MixOrder()

    # move model to GPU
    model.cuda()

    # print the model/summary
    print(model)
