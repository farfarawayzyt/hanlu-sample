# @author : ThinkPad 
# @date : 2023/2/11
import wandb
import numpy as np

from collections import defaultdict
from tqdm import tqdm

ENTITY = 'erichan'
PROJECT = 'open-set-ssl-ICCV'
METRIC_NAME = 'test_accuracy'

api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}")

for run in tqdm(runs):
    if METRIC_NAME in run.history():
        values = run.history()[METRIC_NAME].values
        values = values[~np.isnan(values)]
        print(values.shape)
        run.summary[f"{METRIC_NAME}_max"] = np.max(values)
        run.summary[f"{METRIC_NAME}_min"] = np.min(values)
        run.summary[f"{METRIC_NAME}_std"] = np.std(values)
        run.summary.update()