import hydra
import ee
from download.gedi import GEDI
import json
from utils import setup_default_logging

logger = setup_default_logging('logs', 'GEDI')

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    key = json.load(open(cfg.keyFile))
    credentials = ee.ServiceAccountCredentials(key['client_email'], cfg.keyFile)
    ee.Initialize(credentials)
    gedi = GEDI(**cfg.init)

    if cfg.task == 'download':
        gedi.download(**cfg.download)
    elif cfg.task == 'visualize':
        gedi.getSampleTable(plot=True)
    else:
        logger.error(f'Task not recognized.\nreceived: {cfg.task} \nexpected: download or visualize')

if __name__ == "__main__":
    main()
