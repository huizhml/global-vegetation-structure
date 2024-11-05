from typing import Any, List, Optional, Mapping
import wandb
import lightning.pytorch.loggers.wandb as wb
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


class WandbLogger(wb.WandbLogger):

    def __init__(self,
                 tags: str = None,
                 artifact_path: str = None,
                 **kwargs: Any) -> None:
        self.artifact_path = artifact_path
        if tags is not None:
            tags = [t.strip() for t in tags.rstrip(',').split(',')]
            kwargs.update({'tags': tags})
        super().__init__(**kwargs)
        self.experiment # explicitly call to check if wandb is initialized
        wandb.define_metric('prediction/rh')
        wandb.define_metric('prediction/*', step_metric='prediction/rh')

