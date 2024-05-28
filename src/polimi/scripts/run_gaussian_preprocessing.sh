python ~/RecSysChallenge2024/src/polimi/scripts/gaussian_preprocessing.py \
    -output_dir /home/ubuntu/experiments \
    -dataset_path /home/ubuntu/experiments/subsample_train_small_new \
    -dataset_type train \
    -numerical_transform yeo-johnson \
    --fit