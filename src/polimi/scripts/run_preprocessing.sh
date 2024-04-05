# prepare training dataset
python ./polimi/scripts/preprocessing.py \
    -output_dir /media/disk1/recsys-challenge-2024/experiments \
    -dataset_path /media/disk1/recsys-challenge-2024/dataset/ebnerd_large \
    -dataset_type train
wait 10
# prepare validation dataset
python ./polimi/scripts/preprocessing.py \
    -output_dir /media/disk1/recsys-challenge-2024/experiments \
    -dataset_path /media/disk1/recsys-challenge-2024/dataset/ebnerd_large \
    -dataset_type validation