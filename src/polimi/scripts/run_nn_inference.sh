# evaluating a model
python ~/RecSysChallenge2024/src/polimi/scripts/nn_inference.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_test_new \
    -model_path /mnt/ebs_volume/models/mlp_new_features_trial_66 \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/mlp_tuning_new_trial_66.json \
    -batch_size 5096 \
    --submit \
    -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet
    #--eval \