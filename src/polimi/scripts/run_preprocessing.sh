# prepare training dataset
python /home/francesco/CHALLENGE/RecSysChallenge2024/src/polimi/scripts/batch_preprocessing.py \
    -output_dir /home/francesco/CHALLENGE/experiments \
    -dataset_path /home/francesco/CHALLENGE/dataset/ebnerd_small \
    -dataset_type train \
    -urm_path /home/francesco/CHALLENGE/recsys2024/urm/ner/small \
    -preprocessing_version 147f

wait 10



### devo testare che la mia 147f
# funzioni batch e no urm e models -> 
# funzioni batch con urm e models
# funzioni prepro e no urm e models
# funzioni prepro e urm e models





