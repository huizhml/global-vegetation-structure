from omegaconf import DictConfig, ListConfig

def resolve_args(args):
    """
    Recursively resolve string formatting in DictConfig, dicts, and lists.
    """

    def _resolve(value, context):
        # Resolve strings with format placeholders
        if isinstance(value, str) and '{' in value:
            return value.format(**context)

        # Recurse into dict-like
        if isinstance(value, (DictConfig, dict)):
            return {k: _resolve(v, context) for k, v in value.items()}

        # Recurse into list-like
        if isinstance(value, (ListConfig, list)):
            return [_resolve(v, context) for v in value]

        return value

    resolved = _resolve(args, args)

    # If original is DictConfig, update in place to preserve Hydra behavior
    if isinstance(args, DictConfig):
        for k, v in resolved.items():
            args[k] = v
        return args

    return resolved
