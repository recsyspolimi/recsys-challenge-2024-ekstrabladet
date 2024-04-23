python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /mnt/ebs_volume/recsys2024/urm \
    -dataset_type small \
    -dataset_path /mnt/ebs_volume/recsys2024/dataset \
    -urm_split train \
    -urm_type ner

wait 10

python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /mnt/ebs_volume/recsys2024/urm \
    -dataset_type small \
    -dataset_path /mnt/ebs_volume/recsys2024/dataset \
    -urm_split validation \
    -urm_type ner

wait 10

python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /mnt/ebs_volume/recsys2024/urm \
    -dataset_type small \
    -dataset_path /mnt/ebs_volume/recsys2024/dataset \
    -urm_split test \
    -urm_type ner

wait 10
