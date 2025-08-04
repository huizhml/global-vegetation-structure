import torch
from typing import Any, List, Union
from lightning.pytorch.callbacks.callback import Callback
import modelopt.torch.opt as mto
import pandas as pd
from lightning.pytorch.utilities.rank_zero import rank_zero_only

class OnnxExportCallback(Callback):

    def __init__(self,
                 
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        # save the quantized model
        model_path = f'checkpoints/fake_quantized_model_finetuned{trainer.max_epochs}_epochs_{trainer.logger._experiment.id}.pth'
        mto.save(pl_module, model_path)
        print(f'saved quantized model to {model_path}')
        onnx_path = f'checkpoints/fake_quantized_model_finetuned{trainer.max_epochs}_epochs_{trainer.logger._experiment.id}.onnx'
        input_tensor = torch.randn(1, 12, 544, 544).to('cuda')
        import ipdb; ipdb.set_trace()
        torch.onnx.export(
            pl_module, input_tensor,
            onnx_path, export_params=True, opset_version=13, do_constant_folding=True, input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}})
        print(f'saved quantized model to {onnx_path}')
