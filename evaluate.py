import wandb
from pathlib import Path
from ffcv.loader import Loader, OrderOption
from models.unet import UNet

val_fp = Path.home() / 'data/GEDI/train_subsets/debug0_filtered_v1.beton'
run_ids = ['4fd8j93r']
dataloaders = Loader(val_fp, batch_size=4096, os_cache=False, order=OrderOption.SEQUENTIAL, distributed=False, batches_ahead=3, num_workers=4)
run = wandb.init(project='global-vegetation-structure', name='test')
# TODO: log config
models = []
for run_id in run_ids:
    artifact = run.use_artifact(f'model-{run_id}:best', type='model')
    artifact_fp = artifact.file()
    import ipdb; ipdb.set_trace()
    model = UNet.load_from_checkpoint(artifact_fp)
    
    model.eval()
    models.append(model)
    

for batch in dataloaders:
    for model in models:
        x, y = batch
        y_hat = model(x)
        loss = model.loss_fc(y_hat, y)
        print(loss)
        


    
