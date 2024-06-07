echo "Training MLP on train"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_training.py \
    -output_dir /mnt/ebs_volume/models \
    -dataset_path /mnt/ebs_volume/experiments/subsample_train_new_yeo_johnson_fit \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/mlp_tuning_new_trial_208.json \
    -model_name mlp_new_trial_208_train

wait 10

echo "Submitting MLP trained only on train"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_inference.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_test_new_yeo_johnson_transform_train \
    -model_path /mnt/ebs_volume/models/mlp_new_trial_208_train \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/mlp_tuning_new_trial_208.json \
    -batch_size 20384 \
    --submit \
    -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet

wait 10

echo "Training GANDALF on train"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_training.py \
    -output_dir /mnt/ebs_volume/models \
    -dataset_path /mnt/ebs_volume/experiments/subsample_train_new_yeo_johnson_fit \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/gandalf_tuning_new_trial_130.json \
    -model_name gandalf_new_trial_130_train

wait 10

echo "Submitting GANDALF trained only on train"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_inference.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_test_new_yeo_johnson_transform_train \
    -model_path /mnt/ebs_volume/models/gandalf_new_trial_130_train \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/gandalf_tuning_new_trial_130.json \
    -batch_size 20384 \
    --submit \
    -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet

wait 10

echo "Training deep&cross on train"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_training.py \
    -output_dir /mnt/ebs_volume/models \
    -dataset_path /mnt/ebs_volume/experiments/subsample_train_new_yeo_johnson_fit \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/deep_cross_tuning_new_trial_67.json \
    -model_name deep_cross_tuning_new_trial_67_train

wait 10

echo "Submitting deep&cross trained only on train"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_inference.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_test_new_yeo_johnson_transform_train \
    -model_path /mnt/ebs_volume/models/deep_cross_tuning_new_trial_67_train \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/deep_cross_tuning_new_trial_67.json \
    -batch_size 20384 \
    --submit \
    -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet

wait 10

echo "Evaluating MLP on val"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_inference.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_validation_new_yeo_johnson_transform_train \
    -model_path /mnt/ebs_volume/models/mlp_new_trial_208_train \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/mlp_tuning_new_trial_208.json \
    -batch_size 20384 \
    --eval \
    -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet

wait 10

echo "Evaluating GANDALF on val"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_inference.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_validation_new_yeo_johnson_transform_train \
    -model_path /mnt/ebs_volume/models/gandalf_new_trial_130_train \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/gandalf_tuning_new_trial_130.json \
    -batch_size 20384 \
    --eval \
    -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet

wait 10

echo "Evaluating deep&cross on val"
/home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_inference.py \
    -output_dir ~/experiments \
    -dataset_path /mnt/ebs_volume/experiments/preprocessing_validation_new_yeo_johnson_transform_train \
    -model_path /mnt/ebs_volume/models/deep_cross_tuning_new_trial_67_train \
    -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/deep_cross_tuning_new_trial_67.json \
    -batch_size 20384 \
    --eval \
    -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet

# wait 10

# echo "Training MLP on train+val"
# /home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_training.py \
#     -output_dir /mnt/ebs_volume/models \
#     -dataset_path /mnt/ebs_volume/experiments/subsample_complete_new_yeo_johnson_fit \
#     -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/mlp_tuning_new_trial_208.json \
#     -model_name mlp_new_trial_208_train_val

# wait 10

# echo "Submitting MLP trained on complete dataset"
# /home/ubuntu/RecSysChallenge2024/.venv/bin/python /home/ubuntu/RecSysChallenge2024/src/polimi/scripts/nn_inference.py \
#     -output_dir ~/experiments \
#     -dataset_path /mnt/ebs_volume/experiments/preprocessing_test_new_yeo_johnson_transform_train_val \
#     -model_path /mnt/ebs_volume/models/mlp_new_trial_208_train_val \
#     -params_file /home/ubuntu/RecSysChallenge2024/configuration_files/mlp_tuning_new_trial_208.json \
#     -batch_size 20384 \
#     --submit \
#     -behaviors_path /home/ubuntu/dataset/ebnerd_testset/test/behaviors.parquet