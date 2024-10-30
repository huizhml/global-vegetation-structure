from typing import Any, List, Optional, Mapping
import wandb
import lightning.pytorch.loggers.wandb as wb
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class WandbLogger(wb.WandbLogger):

    def __init__(self,
                 tags: str = None,
                 artifact_path: str = None,
                 **kwargs: Any) -> None:
        self.artifact_path = artifact_path
        print('******** allow_val_change: ',
              kwargs.get('allow_val_change', None))
        if kwargs.get('resume', None):
            kwargs = {
                'id': kwargs['id'],
                'project': kwargs['project'],
                'job_type': kwargs['project'],
                'resume': 'allow'
            }
        elif tags is not None:
            tags = [t.strip() for t in tags.rstrip(',').split(',')]
            kwargs.update({'tags': tags})
        super().__init__(**kwargs)
        self.experiment # explicitly call to check if wandb is initialized
        wandb.define_metric('prediction/rh')
        wandb.define_metric('prediction/*', step_metric='prediction/rh')

