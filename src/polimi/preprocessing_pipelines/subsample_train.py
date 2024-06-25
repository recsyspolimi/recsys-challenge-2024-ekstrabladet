import os
import json
import argparse
from polimi.utils._catboost import subsample_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for building emotions embeddingd")
    parser.add_argument("-output_dir", default="../../experiments/", type=str,
                        help="The directory where the sumbsampled dataset will be placed")
    parser.add_argument("-dataset_dir", default=None, type=str, required=True,
                        help="Directory of dataframe that has to be subsampled")
    parser.add_argument("-original_path", default=None, type=str, required=True,
                        help="Directory of the starting behavior dataframe (no preprocessing)")
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_dir
    ORIGINAL_PATH = args.original_path
    
    os.makedirs(OUTPUT_DIR)
    
    output_path = OUTPUT_DIR + '/train_ds.parquet'
    input_path = DATASET_DIR + '/train_ds.parquet'
    
    subsample_dataset(ORIGINAL_PATH, 
                      input_path,
                      output_path)
    
    data_info_path = os.path.join(DATASET_DIR, 'data_info.json')
    with open(data_info_path, "r") as data_info:
        to_save = json.load(data_info)
    
    destination = os.path.join(OUTPUT_DIR, 'data_info.json')
    with open(destination, "w") as to:
        json.dump(to_save, to)
    