# evaluating a model
python ./polimi/scripts/inference.py \
    -output_dir /media/disk1/recsys-challenge-2024/experiments \
    -dataset_path /media/disk1/recsys-challenge-2024/experiments/preprocessing_validation_2024-04-05_00-28-29 \
    -model_path /media/disk1/recsys-challenge-2024/experiments/Lightgbm_training_ \
    -eval

wait 10

# producing a submission
python ./polimi/scripts/inference.py \
    -output_dir /media/disk1/recsys-challenge-2024/experiments \
    -dataset_path /media/disk1/recsys-challenge-2024/experiments/preprocessing_validation_2024-04-05_00-28-29 \
    -model_path /media/disk1/recsys-challenge-2024/experiments/Lightgbm_training_ \
    -submit \
    -behaviors_path /media/disk1/recsys-challenge-2024/dataset/ebnerd_testset/test/behaviors.parquet
