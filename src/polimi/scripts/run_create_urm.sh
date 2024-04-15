python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /home/ubuntu/urm \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_large \
    -urm_split train \
    -urm_type ner

wait 10


python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /home/ubuntu/urm \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_large \
    -urm_split validation \
    -urm_type ner

wait 10


python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /home/ubuntu/urm \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset/ebnerd_large \
    -urm_split test \
    -urm_type ner

