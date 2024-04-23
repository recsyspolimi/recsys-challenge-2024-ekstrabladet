python ~/RecSysChallenge2024/src/polimi/scripts/urm_tuning.py \
    -output_dir /mnt/ebs_volume/recsys2024/urm/tuning \
    -urm_folder /mnt/ebs_volume/recsys2024/urm/ner/small \
    -n_trials 1000 \
    -study_name PureSVDRecommender-ner-small-ndcg10_new \
    -storage mysql+pymysql://admin:MLZwrgaib8iha7DU9jgP@recsys2024-db.crio26omekmi.eu-west-1.rds.amazonaws.com/recsys2024 \
    -model_name PureSVDRecommender \
    -metric NDCG
