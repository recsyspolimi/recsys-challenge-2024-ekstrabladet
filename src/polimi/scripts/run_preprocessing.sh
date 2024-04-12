# prepare training dataset
python ~/RecSysChallenge2024/src/polimi/scripts/preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_large \
    -dataset_type train

wait 10

# prepare test dataset
python ~/RecSysChallenge2024/src/polimi/scripts/preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_testset \
    -dataset_type test

wait 10

# prepare validation dataset
python ~/RecSysChallenge2024/src/polimi/scripts/preprocessing.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_large \
    -dataset_type validation

