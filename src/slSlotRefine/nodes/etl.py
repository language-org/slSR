
import os

import yaml


def init_data_paths(data):

    # print
    print_dataset(data)

    # get dataset (e.g., atis or snips)
    dataset = data.arg.dataset
    
    # get dataset path
    data_path = data.catalog["input"]["path"]
    inference_path = data.catalog["inference"]["load_path"]

    # training dataset
    data.full_train_path = os.path.join(data_path, dataset, data.arg.train_data_path)
    
    # labelled test dataset
    data.full_test_path = os.path.join(data_path, dataset, data.arg.test_data_path)
    
    data.full_test_write_path = inference_path

    # labelled validation dataset
    data.full_valid_path = os.path.join(data_path, dataset, data.arg.valid_data_path)
    return data


def print_dataset(data):
    
    # full path to data will be: ./data + dataset + train/test/valid
    # case
    if data.arg.dataset is None:
        raise ValueError("name of dataset can not be None")
        exit(1)
    elif data.arg.dataset == "snips":
        print("use snips dataset")
    elif data.arg.dataset == "atis":
        print("use atis dataset")
    else:
        print("use own dataset: ", data.arg.dataset)
