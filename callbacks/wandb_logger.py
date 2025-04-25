from typing import Any, List, Optional, Mapping, Tuple
import wandb
from pathlib import Path
import lightning.pytorch.loggers.wandb as wb
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import Checkpoint
from torch import Tensor

def _scan_checkpoints(checkpoint_callback: Checkpoint, logged_model_time: dict) -> List[Tuple[float, str, float, str]]:
    """Return the checkpoints to be logged.

    Args:
        checkpoint_callback: Checkpoint callback reference.
        logged_model_time: dictionary containing the logged model times.

    """
    # get checkpoints to be saved with associated score
    checkpoints = {}
    if hasattr(checkpoint_callback, "last_model_path") and checkpoint_callback.current_score is not None: ##! Changed here, model checkpoint has current_score and best_model_score initialized as none
        checkpoints[checkpoint_callback.last_model_path] = (checkpoint_callback.current_score, "latest")

    if hasattr(checkpoint_callback, "best_model_path") and checkpoint_callback.best_model_score is not None:
        checkpoints[checkpoint_callback.best_model_path] = (checkpoint_callback.best_model_score, "best")

    if hasattr(checkpoint_callback, "best_k_models"):
        for key, value in checkpoint_callback.best_k_models.items():
            checkpoints[key] = (value, "best_k")

    checkpoints = sorted(
        (Path(p).stat().st_mtime, p, s, tag) for p, (s, tag) in checkpoints.items() if Path(p).is_file()
    )
    checkpoints = [c for c in checkpoints if c[1] not in logged_model_time] ##! Changed here, no need to log every intermediate checkpoint, e.g, last-v100.ckpt, which is used to keep track of the latest checkpoint

    return checkpoints

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
        # wandb.define_metric('prediction/rh')
        # wandb.define_metric('prediction/*', step_metric='prediction/rh')


    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        import wandb

        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p),
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"
            artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t

    # @override
    # def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
    #     # log checkpoints as artifacts
    #     log = self._log_model == "all" or (self._log_model is True and checkpoint_callback.save_top_k == -1) or
    #     if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
    #         self._scan_and_log_checkpoints(checkpoint_callback)
    #     elif self._log_model:
    #         self._checkpoint_callback = checkpoint_callback