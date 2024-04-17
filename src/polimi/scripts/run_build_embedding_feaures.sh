# build features
python ~/RecSysChallenge2024/src/polimi/scripts/build_embedding_features.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_large \
    -test_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_testset \
    -feature_name emotions_user_item_distance \
    -emdeddings_file /home/ubuntu/dataset/articles_emotions.parquet 

wait 10
