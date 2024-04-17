# build features
python ~/RecSysChallenge2024/src/polimi/scripts/build_embedding_features.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_large \
    -test_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_testset \
    -feature_name word2vec_user_item_distance \
    -emdeddings_file /mnt/ebs_volume/recsys_challenge/dataset/Ekstra_Bladet_word2vec/document_vector.parquet

wait 10

python ~/RecSysChallenge2024/src/polimi/scripts/build_embedding_features.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_large \
    -test_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_testset \
    -feature_name contrastive_user_item_distance \
    -emdeddings_file /mnt/ebs_volume/recsys_challenge/dataset/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet

wait 10

python ~/RecSysChallenge2024/src/polimi/scripts/build_embedding_features.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_large \
    -test_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_testset \
    -feature_name roberta_user_item_distance \
    -emdeddings_file /mnt/ebs_volume/recsys_challenge/dataset/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet

wait 10