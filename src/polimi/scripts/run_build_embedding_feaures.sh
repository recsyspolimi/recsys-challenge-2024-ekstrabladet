python ~/RecSysChallenge2024/src/polimi/scripts/build_embedding_features.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_large \
    -test_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_testset \
    -feature_name word2vec_user_item_distance \
    -dataset_type /mnt/ebs_volume/recsys_challenge/dataset/Ekstra_Bladet_word2vec.document_vector.parquet 

wait 10
