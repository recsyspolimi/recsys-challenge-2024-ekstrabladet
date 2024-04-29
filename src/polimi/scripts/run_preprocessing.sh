# prepare training dataset
python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path /home/ubuntu/dataset/ebnerd_large \
    -dataset_type train \
    -previous_version /home/ubuntu/experiments/preprocessing_train_2024-04-22_13-28-45/train_ds.parquet

wait 10

# prepare validation dataset
python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path /home/ubuntu/dataset/ebnerd_large \
    -dataset_type validation \
    -previous_version /home/ubuntu/experiments/preprocessing_validation_2024-04-23_08-38-39/validation_ds.parquet

wait 10
# prepare test dataset
python ~/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path ~/dataset/ebnerd_testset \
    -dataset_type test \
    -previous_version /home/ubuntu/experiments/preprocessing_test_2024-04-25_10-03-50

wait 10




