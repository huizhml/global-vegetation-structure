from dataclasses import field, make_dataclass
from pathlib import Path
from typing import Any, List, Union
import yaml
from omegaconf import MISSING, OmegaConf
from hydra.core.config_store import ConfigStore


def _split_shared_and_ops(d: dict) -> tuple[dict, dict]:
    """Split a YAML mapping into (shared_scalars, op_dicts)."""
    shared = {k: v for k, v in d.items() if not isinstance(v, dict)}
    ops = {k: v for k, v in d.items() if isinstance(v, dict)}
    return shared, ops


def register(
    yaml_path: Union[str, Path],
    section: str,
    default_run: str,
    group: str = 'run',
) -> list[str]:
    """One-shot registration for a per-module entrypoint.

    Reads the YAML at `yaml_path`, picks `section`, registers:
      1. `base_config` (RunConfig dataclass) with shared values as top-level
         fields, so they're accessible at the config root for `${name}`
         interpolation.
      2. Every op under `section` into the ConfigStore group `group`.

    YAML layout expected:
        root_data_dir: ~/data/gvs        # file-level shared values (optional)
        product_dir: ~/data/gvs/product

        section_a:
          shared_in_section: foo         # section-level shared values (optional)
          op_1:                          # an op (dict)
            _target_: ...
            field_a: ${root_data_dir}/x  # absolute ref to file-level shared
            field_b: ${.field_a}/y       # relative ref to sibling field
          op_2:
            ...
        section_b:
          ...

    Interpolation rules (OmegaConf):
      - `${name}`   absolute from config root  -> resolves cfg.<name>
      - `${.name}`  relative to current node    -> resolves a sibling field

    Section-level shared overrides file-level. Per-op fields override shared.

    Returns the list of registered op names.
    """
    yaml_path = Path(yaml_path).expanduser()
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    file_shared, _ = _split_shared_and_ops(data)
    if section not in data or not isinstance(data[section], dict):
        raise KeyError(f"section '{section}' not found in {yaml_path}")
    section_shared, ops = _split_shared_and_ops(data[section])
    shared = {**file_shared, **section_shared}

    _register_base(default_run=default_run, shared=shared, group=group, section=section)

    cs = ConfigStore.instance()
    for name, cfg in ops.items():
        cs.store(group=group, name=name, node=OmegaConf.create(cfg))
    return list(ops.keys())


def _register_base(default_run: str, shared: dict, group: str = 'run',
                   section: str | None = None) -> None:
    """Build a RunConfig dataclass with `defaults`, `run`, and `shared` fields
    at the top level, and register it as `base_config`.

    Top-level shared values are required to live at the config root so that
    ops can reference them via absolute `${name}` interpolation. `section` (the
    registering module's section) is stored too, so the runner can name the
    results index `<section>/<op>`.
    """
    defaults_list = [{group: default_run}, '_self_']
    cls_fields = [
        ('defaults', List[Any], field(default_factory=lambda: list(defaults_list))),
        ('run', Any, field(default=MISSING)),
    ]
    if section is not None:
        cls_fields.append(('section', str, field(default=section)))
    for k, v in shared.items():
        if isinstance(v, (list, dict)):
            cls_fields.append((k, type(v), field(default_factory=lambda v=v: v)))
        else:
            t = type(v) if v is not None else Any
            cls_fields.append((k, t, field(default=v)))

    RunConfig = make_dataclass('RunConfig', cls_fields)
    ConfigStore.instance().store(name='base_config', node=RunConfig)
