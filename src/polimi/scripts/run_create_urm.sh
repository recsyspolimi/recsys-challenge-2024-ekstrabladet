python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /mnt/ebs_volume_2/recsys2024/urm \
    -dataset_type large \
    -dataset_path /home/ubuntu/dataset \
    -urm_split train \
    -urm_type recsys

python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /mnt/ebs_volume_2/recsys2024/urm \
    -dataset_type large \
    -dataset_path /home/ubuntu/dataset \
    -urm_split validation \
    -urm_type recsys

python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir /mnt/ebs_volume_2/recsys2024/urm \
    -dataset_type testset \
    -dataset_path /home/ubuntu/dataset \
    -urm_split test \
    -urm_type recsys