python ~/RecSysChallenge2024/src/polimi/scripts/train_recommenders.py \
    -urm_path /mnt/ebs_volume_2/recsys2024/urm/recsys/large \
    -urm_split train\
    -dataset_type large\
    -urm_type recsys\
    -models all\
    -output_dir /mnt/ebs_volume_2/recsys2024/algo/\

python ~/RecSysChallenge2024/src/polimi/scripts/train_recommenders.py \
    -urm_path /mnt/ebs_volume_2/recsys2024/urm/recsys/large \
    -urm_split validation\
    -dataset_type large\
    -urm_type recsys\
    -models all\
    -output_dir /mnt/ebs_volume_2/recsys2024/algo/\

python ~/RecSysChallenge2024/src/polimi/scripts/train_recommenders.py \
    -urm_path /mnt/ebs_volume_2/recsys2024/urm/recsys/testset \
    -urm_split test\
    -dataset_type testset\
    -urm_type recsys\
    -models all\
    -output_dir /mnt/ebs_volume_2/recsys2024/algo/\


    


    
