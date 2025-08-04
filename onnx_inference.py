import hydra
import psutil
import torch
import numpy as np
import onnxruntime
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
from datasets._zarr_dataset_deploy import DeployDataModel
from datasets.transforms import Normalize


def prepare_batch(sample, transform, feed_latlon=True):
    if feed_latlon:
        x = transform(sample[0])
        x = torch.cat([x, sample[-1]], dim=1)
    else:
        x = transform(sample[0])
    return x

@hydra.main(config_name="deploy", config_path="config", version_base="1.2")
def main(cfg):
    datamodule = DeployDataModel(**cfg.data.init_args)
    pred_dataloader = datamodule.predict_dataloader()    
    assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
    sess_options = onnxruntime.SessionOptions()
    sess_options.optimized_model_filepath = str(Path(cfg.model.onnx_model_path).expanduser())
    sess_options.enable_profiling = True
    sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    
    providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                    "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
    
    session = onnxruntime.InferenceSession(cfg.model.onnx_model_path, sess_options, providers=providers)
    transform = Normalize()
    binding = session.io_binding()
     
    for batch_idx, sample in tqdm(enumerate(pred_dataloader)):
        x = prepare_batch(sample, transform)
        x_tensor = x.contiguous().to(torch.float16).cuda() # (2, 15, 544, 544)
        binding.bind_input(
            'input',
            device_type='cuda',
            device_id=0,
            element_type=np.float16, 
            shape=tuple(x_tensor.shape),
            buffer_ptr=x_tensor.data_ptr(),
        )
        y_shape = (20, 315, 544, 544)
        y_tensor = torch.empty(y_shape, dtype=torch.float16, device='cuda').contiguous()
        binding.bind_output(
            'output',
            device_type='cuda',
            device_id=0,
            element_type=np.float16,
            shape=y_shape,
            buffer_ptr=None,
        )
        # y_tensor = session.run(None, ort_inputs)
        session.run_with_iobinding(binding)
    profile = session.end_profiling()
    print('profile written to ', profile)
    
if __name__ == '__main__':
    main()