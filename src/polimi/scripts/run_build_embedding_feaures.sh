python3 ~/RecSysChallenge2024/src/polimi/scripts/build_embedding_features.py \
    -output_dir /home/ubuntu/experiments \
    -dataset_path /home/ubuntu/dataset/ebnerd_large \
    -feature_name bert_user_item_distance \
    -emdeddings_file /home/ubuntu/dataset/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet \
    -test_path /home/ubuntu/dataset/ebnerd_testset
wait 10