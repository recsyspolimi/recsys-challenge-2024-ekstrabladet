# prepare training dataset
python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path ~/dataset/ebnerd_large \
    -dataset_type train \
    -preprocessing_version 94f
wait 10

# prepare validation dataset
python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path ~/dataset/ebnerd_large \
    -dataset_type validation \
    -preprocessing_version 94fun
    
# prepare test dataset
python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path ~/dataset/ebnerd_testset \
    -dataset_type test \
    -preprocessing_version 94f

wait 10




