import onnx
from onnxconverter_common import float16

import wandb
import torch
from tqdm import tqdm
from models.xception_s2 import XceptionS2MixOrder
from models.modules.xception_blocks import DoubleSepConvBlock, SepConvBlockSkip, DoubleConvBlock



model = XceptionS2MixOrder(
    in_channels=15,
    out_channels=315,
    entry_block_filters=[128, 256],
    long_skip=True,
    activation_layer=torch.nn.ReLU(),
    norm_layer='torch.nn.BatchNorm2d',
    sepconv_block='models.modules.xception_blocks.DoubleSepConvBlock',
    num_sepconv_blocks=3,
    num_sepconv_filters=512,
    mid_block='models.modules.xception_blocks.SepConvBlockSkip',
    nonlin_block='models.modules.xception_blocks.DoubleConvBlock',
    num_nonlin_blocks=4,
    nonlinear_order='inbetween',
)

model_alias = 'best'
wandb_project = 'global-vegetation-structure-v1'
run_id = 'cg11fpjr'
api = wandb.Api()
run_path = f"{wandb_project}/{run_id}"
run = api.run(run_path)

artifacts = run.logged_artifacts()
artifacts = [artifact for artifact in artifacts if artifact.type == 'model']
if isinstance(model_alias, str):
    artifact = [art for art in artifacts if model_alias in art.aliases][0]
else:
    artifact = artifacts[model_alias]
# artifact = run_.use_artifact(f'model-{run_id}:best', type='model')
ckpt_path = artifact.file()

checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model = model.to('cuda')
model = model.eval()
with torch.no_grad():
    torch.onnx.export(model, 
                  torch.randn(10, 15, 544, 544).to('cuda'), 
                  f"checkpoints/model_{run_id}.onnx", verbose=True,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})




model = onnx.load(f"checkpoints/model_{run_id}.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, f"checkpoints/model_{run_id}_fp16.onnx")
import ipdb; ipdb.set_trace()



