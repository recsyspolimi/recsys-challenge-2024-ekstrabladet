python ~/RecSysChallenge2024/src/polimi/scripts/build_recsys_scores_features.py \
    -dataset_split train \
    -dataset_type large \
    -urm_type recsys \
    -base_path /mnt/ebs_volume_2/recsys2024/\
    -output_dir /mnt/ebs_volume_2/recsys2024/features/\

python ~/RecSysChallenge2024/src/polimi/scripts/build_recsys_scores_features.py \
    -dataset_split validation \
    -dataset_type large \
    -urm_type recsys \
    -base_path /mnt/ebs_volume_2/recsys2024/\
    -output_dir /mnt/ebs_volume_2/recsys2024/features/\

python ~/RecSysChallenge2024/src/polimi/scripts/build_recsys_scores_features.py \
    -dataset_split test \
    -dataset_type testset \
    -urm_type recsys \
    -base_path /mnt/ebs_volume_2/recsys2024/\
    -output_dir /mnt/ebs_volume_2/recsys2024/features/\
    
    


    
