echo "Fit yeo johnson on the training set"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/gaussian_preprocessing.py \
    -output_dir /mnt/ebs_volume/experiments/subsample_train_new_yeo_johnoson_fit \
    -dataset_path /mnt/ebs_volume/experiments/subsample_train_new \
    -dataset_type train \
    -numerical_transform yeo-johnson \
    --fit

wait 10

echo "Transform test data for submissions training only on train"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/gaussian_preprocessing.py \
    -output_dir /mnt/ebs_volume/experiments/preprocessing_test_new_yeo_johnoson_transform_train \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_test_new \
    -dataset_type test \
    -numerical_transform yeo-johnson \
    -load_path /mnt/ebs_volume/experiments/subsample_train_new_yeo_johnoson_fit/numerical_transformer.joblib \
    # -transform_batch_size 100000

wait 10

echo "Transform validation data to generate the predictions for the level 2 model"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/gaussian_preprocessing.py \
    -output_dir /mnt/ebs_volume/experiments/preprocessing_validation_new_yeo_johnoson_transform_train \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_validation_new \
    -dataset_type validation \
    -numerical_transform yeo-johnson \
    -load_path /mnt/ebs_volume/experiments/subsample_train_new_yeo_johnoson_fit/numerical_transformer.joblib

wait 10

echo "Fit yeo johnson on the complete dataset"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/gaussian_preprocessing.py \
    -output_dir /mnt/ebs_volume/experiments/subsample_complete_new_yeo_johnoson_fit \
    -dataset_path /mnt/ebs_volume/experiments/subsample_complete_new \
    -dataset_type train \
    -numerical_transform yeo-johnson \
    --fit

wait 10

echo "Transform test data"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/gaussian_preprocessing.py \
    -output_dir /mnt/ebs_volume/experiments/preprocessing_test_new_yeo_johnoson_transform_train_val \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_test_new \
    -dataset_type test \
    -numerical_transform yeo-johnson \
    -load_path /mnt/ebs_volume/experiments/subsample_complete_new_yeo_johnoson_fit/numerical_transformer.joblib