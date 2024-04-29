# prepare training dataset
python /Users/lorecampa/Desktop/Projects/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir /Users/lorecampa/Desktop/Projects/RecSysChallenge2024/experiments \
    -dataset_path /Users/lorecampa/Desktop/Projects/RecSysChallenge2024/dataset/ebnerd_small \
    -dataset_type train \
    -preprocessing_version 94f

wait 10

# prepare validation dataset
python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path ~/dataset/ebnerd_small \
    -dataset_type validation 
    
# prepare test dataset
python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path ~/dataset/ebnerd_testset \
    -dataset_type test \
    -preprocessing_version 94f

wait 10




