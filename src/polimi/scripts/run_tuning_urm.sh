python ~/RecSysChallenge2024/src/polimi/scripts/urm_tuning.py \
    -output_dir /mnt/ebs_volume/recsys2024/urm/tuning \
    -urm_folder /mnt/ebs_volume/recsys2024/urm/ner/ebnerd_small/train \
    -n_trials 1000 \
    -study_name PureSVDItemRecommender-ner-small-ndcg100_v2 \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
    -model_name PureSVDRecommender \
    -metric NDCG \
    -cutoff 100
