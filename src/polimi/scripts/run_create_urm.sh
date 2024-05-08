python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir ~/urm \
    -dataset_type small \
    -dataset_path ~/dataset \
    -urm_split train \
    -urm_type ner

python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir ~/urm \
    -dataset_type small \
    -dataset_path ~/dataset \
    -urm_split validation \
    -urm_type ner


python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir ~/urm  \
    -dataset_type small \
    -dataset_path ~/dataset \
    -urm_split train_val \
    -urm_type ner

python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir ~/urm \
    -dataset_type small \
    -dataset_path ~/dataset \
    -urm_split test \
    -urm_type ner

#testset command, run to create the testset urm with history_train_large + history_val_large + history_testset
python ~/RecSysChallenge2024/src/polimi/scripts/create_urm.py \
    -output_dir ~/urm \
    -dataset_type testset \
    -dataset_path ~/dataset \
    -urm_split train \
    -urm_type ner
