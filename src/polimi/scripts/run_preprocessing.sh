# prepare training dataset
python ~/RecSysChallenge2024/src/polimi/scripts/preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path /home/ubuntu/dataset/ebnerd_small \
    -dataset_type train \
    -previous_version /home/ubuntu/experiments/preprocessing_train_small_127/train_ds.parquet \
    -urm_path /home/ubuntu/dataset/urm/urm/ner/small \
    -preprocessing_version 142f

wait 10

# prepare validation dataset
python ~/RecSysChallenge2024/src/polimi/scripts/preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path /home/ubuntu/dataset/ebnerd_small \
    -dataset_type validation \
    -previous_version /home/ubuntu/experiments/preprocessing_validation_small_127/validation_ds.parquet \
    -urm_path /home/ubuntu/dataset/urm/urm/ner/small \
    -preprocessing_version 142f

wait 10
# prepare test dataset
python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path ~/dataset/ebnerd_testset \
    -dataset_type test \
    -previous_version /home/ubuntu/experiments/preprocessing_test_2024-04-25_10-03-50

wait 10




