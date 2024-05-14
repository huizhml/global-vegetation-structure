from typing import Union, List

def check_if_log(current_epoch, log_every: Union[int, List]):
    if isinstance(log_every, int):
        log_epoch = (current_epoch + 1) % log_every == 0
    elif isinstance(log_every, list):
        log_epoch = current_epoch in log_every
    else:
        log_epoch = False
    return log_epoch