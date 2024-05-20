# WARNING
# prima di runnare 147f, assicurarci di aver già:
#   - run_create_urm.sh
#   - run_train_recommenders.sh
# da passare come parametri

python ~/RecSysChallenge2024/src/polimi/scripts/preprocessing_moving_kfold.py \
    -output_dir /home/ubuntu/experiments \
    -dataset_path /home/ubuntu/dataset/ebnerd_small \
    -preprocessing_version 127f \




python /Users/lorecampa/Desktop/Projects/RecSysChallenge2024/src/polimi/scripts/preprocessing_moving_kfold.py \
    -output_dir /Users/lorecampa/Desktop/Projects/RecSysChallenge2024/experiments \
    -dataset_path /Users/lorecampa/Desktop/Projects/RecSysChallenge2024/dataset/ebnerd_demo \
    -preprocessing_version new


