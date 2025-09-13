import os
import sys
if "prompt_expert" not in os.getcwd():
    os.chdir("prompt_expert")
from datasets import load_dataset, list_datasets
import pickle

# CoLA SST MRPC STS QQP MNLI QNLI RTE

dataset_names = ['boolq', 'cb', 'copa', 'multirc', 'rte', 'wic', 'wsc', 'record']

for dataset in dataset_names:
    os.makedirs(f"data/superglue/{dataset}", exist_ok=True)
    data = load_dataset('super_glue', dataset, trust_remote_code=True)
    for key in data.keys():
        path = f"data/superglue/{dataset}/{key}.pkl"
        # d = load_dataset('glue', dataset)[key]
        # for x in d:
        #     print(x)
        pickle.dump(data[key], open(path, 'wb'))


