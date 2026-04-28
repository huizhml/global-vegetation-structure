from omegaconf.dictconfig import DictConfig

def resolve_args(args: DictConfig):
    '''
    Resolve the arguments
    Args:
        args: dictionary of arguments
    Returns:
        resolved arguments
    '''
    for key, value in args.items():
        if isinstance(value, str) and '{' in value:
            args[key] = value.format(**args)
    return args


