python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /mnt/ebs_volume/urm \
    -dataset_type large \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset \
    -urm_split train \
    -urm_type ner

wait 10


python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /mnt/ebs_volume/urm \
    -dataset_type large \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset \
    -urm_split validation \
    -urm_type ner

wait 10

python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /mnt/ebs_volume/urm \
    -dataset_type large \
    -dataset_path /mnt/ebs_volume/recsys_challenge/dataset \
    -urm_split test \
    -urm_type ner

wait 10

