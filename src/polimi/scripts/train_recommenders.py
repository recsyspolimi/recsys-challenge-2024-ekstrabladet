import logging
from pathlib import Path
from datetime import datetime
import argparse
import polars as pl
import os
from polimi.utils._urm import build_user_id_mapping, build_ner_mapping, build_ner_urm,build_recsys_urm,build_item_mapping, build_articles_with_processed_ner, compute_sparsity_ratio
from polimi.utils._custom import load_sparse_csr,save_sparse_csr, read_json, ALGORITHMS,algo_dict_ner,algo_dict_recsys
from polimi.utils._urm import train_recommender


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def main(urm_path: Path, models: str, urm_split: str ,output_dir: Path, urm_type:str, dataset_type: str):
    logging.info(f"Loading the URM from {urm_path}")
    URM_train = load_sparse_csr(Path(os.path.join(urm_path, f'URM_{urm_split}.npz')))
    
    output_dir = output_dir.joinpath(urm_type).joinpath(dataset_type).joinpath(urm_split)
    models_list = models.split()
    algo_dict = {}
    
    if urm_type == 'recsys':
        algo_dict=algo_dict_recsys
    else:
        algo_dict=algo_dict_ner
    
    if models_list[0] == 'all':
        for rec,params in algo_dict.items():
            logging.info(f'Training recommender: {rec.RECOMMENDER_NAME}')
            rec_instance = train_recommender(URM_train,recommender=rec,params=params['params'],output_dir=output_dir,file_name=rec.RECOMMENDER_NAME)
            
        
    else:
        for rec in models_list:
            params = algo_dict[ALGORITHMS[rec][0]]['params']
            logging.info(f'Training recommender: {rec}')
            rec_instance = train_recommender(URM_train,recommender=ALGORITHMS[rec][0],params=params,output_dir=output_dir,file_name=rec)
 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training models...")
    parser.add_argument("-urm_path", default=None, type=str, required=True,
                        help="Directory which contains the urm used to train the models")
    parser.add_argument("-models",default='all', type=str, required = True,
                        help="Specify which type of models you want to train, with the names of the models divided with a space. If all run all the with the best params")
    parser.add_argument("-urm_split", choices=['train', 'validation', 'test'], default='train',required=True, type=str,
                        help="Specify the type of URM split: ['train', 'validation', 'test', 'train_val']")
    parser.add_argument("-output_dir", default=None, type=str,required=True,
                        help="The directory where the trained models will be placed")
    parser.add_argument("-urm_type", choices=['ner', 'recsys'],required=True, default='ner', type=str,
                        help="Specify the type of URM: ['ner', 'recsys']")
    parser.add_argument("-dataset_type", choices=['demo', 'small', 'large', 'testset'],required=True, default='small', type=str,
                        help="Specify the type of dataset: ['demo', 'small', 'large']")
  
    args = parser.parse_args()
    URM_PATH = Path(args.urm_path)
    MODELS = args.models
    URM_SPLIT = args.urm_split
    OUTPUT_DIR = Path(args.output_dir)
    URM_TYPE = args.urm_type
    DATASET_TYPE = args.dataset_type
       
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR.joinpath("log.txt")    
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(URM_PATH, MODELS, URM_SPLIT, OUTPUT_DIR, URM_TYPE, DATASET_TYPE)
