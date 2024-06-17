import polars as pl
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import numpy as np
import logging
import random
import os
import json
import joblib
from datetime import datetime
import logging
import gc

from polimi.utils.tf_models.utils.build_sequences import build_history_seq,  build_sequences_cls_iterator, N_CATEGORY, N_SENTIMENT_LABEL, N_SUBCATEGORY, N_TOPICS, N_HOUR_GROUP, N_WEEKDAY
from polimi.utils.tf_models import AttentionHistoryClassificationModel
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from polimi.utils.tf_models.utils import get_simple_decay_scheduler

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(seed)

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f'/home/ubuntu/experiments/rnn_attention_all_{timestamp}'
    os.makedirs(save_dir)
    
    checkpoint_dir = f'{save_dir}/checkpoints'
    os.makedirs(checkpoint_dir)
    
    log_path = os.path.join(save_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))
    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    logging.info(f'Reading dataset')
    
    history_train = pl.read_parquet(f'/home/ubuntu/dataset/ebnerd_small/train/history.parquet')
    n_users = len(history_train)
    history_val = pl.read_parquet(f'/home/ubuntu/dataset/ebnerd_small/validation/history.parquet')
    articles = pl.read_parquet(f'/home/ubuntu/dataset/ebnerd_small/articles.parquet')

    history_seq_train = build_history_seq(history_train, articles)
    history_seq_val = build_history_seq(history_val, articles)
    
    logging.info(f'Built history sequences')
    
    del history_train, history_val, articles
    gc.collect()

    with open('/home/ubuntu/dset_complete/subsample/data_info.json', 'r') as data_info_file:
        data_info = json.load(data_info_file)
        
    logging.info(f'Processing training features')
        
    behaviors = pl.read_parquet('/home/ubuntu/dset_complete/subsample/train_ds.parquet')
    categorical_columns = [c for c in data_info['categorical_columns'] if c in behaviors.columns]
    numerical_columns = [c for c in behaviors.columns if c not in categorical_columns + ['target', 'user_id', 'impression_id', 'article', 'impression_time']]

    behaviors_pandas = behaviors.to_pandas()

    xformer = PowerTransformer()
    behaviors_pandas[numerical_columns] = behaviors_pandas[numerical_columns].replace([-np.inf, np.inf], np.nan).fillna(0)
    behaviors_pandas[numerical_columns] = xformer.fit_transform(behaviors_pandas[numerical_columns]).astype(np.float32)

    vocabulary_sizes = {}
    categories = []
    for f in categorical_columns:
        behaviors_pandas[f] = behaviors_pandas[f].astype(str).fillna('NA')
        categories_train = list(behaviors_pandas[f].unique())
        if 'NA' not in categories_train:
            categories_train.append('NA')
        categories.append(categories_train)
        vocabulary_sizes[f] = len(categories_train)
                
    encoder = OrdinalEncoder(categories=categories)
    behaviors_pandas[categorical_columns] = encoder.fit_transform(behaviors_pandas[categorical_columns]).astype(np.int16)
    behaviors = behaviors.select(['target', 'user_id']).hstack(pl.from_pandas(behaviors_pandas[numerical_columns + categorical_columns]))
    
    del behaviors_pandas
    gc.collect()
    
    logging.info(f'Processing validation features')
    
    behaviors_val = pl.read_parquet('/home/ubuntu/dset_complete/subsample/validation_ds.parquet')
    behaviors_pandas = behaviors_val.to_pandas()

    behaviors_pandas[numerical_columns] = behaviors_pandas[numerical_columns].replace([-np.inf, np.inf], np.nan).fillna(0)
    behaviors_pandas[numerical_columns] = xformer.transform(behaviors_pandas[numerical_columns]).astype(np.float32)

    for i, f in enumerate(categorical_columns):
        behaviors_pandas[f] = behaviors_pandas[f].astype(str).fillna('NA')
        categories_val = list(behaviors_pandas[f].unique())
        unknown_categories = [x for x in categories_val if x not in categories[i]]
        behaviors_pandas[f] = behaviors_pandas[f].replace(list(unknown_categories), 'NA')
    behaviors_pandas[categorical_columns] = encoder.transform(behaviors_pandas[categorical_columns]).astype(np.int16)
    behaviors_val = behaviors_val.select(['target', 'user_id']).hstack(pl.from_pandas(behaviors_pandas[numerical_columns + categorical_columns]))
    
    del behaviors_pandas
    gc.collect()
        
    info = {
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'categories': [categories.tolist() for categories in encoder.categories_],
        'vocabulary_sizes': vocabulary_sizes
    }
    with open(f'{save_dir}/info.json', 'w') as info_file:
        json.dump(info, info_file)
        
    joblib.dump(xformer, f'{save_dir}/power_transformer.joblib')
    joblib.dump(encoder, f'{save_dir}/ordinal_encoder.joblib')
    
    window = 30
    output_signature = (
        {
            'numerical_columns': tf.TensorSpec(shape=(len(numerical_columns),), dtype=tf.float32), # behaviors numerical columns
            **{c: tf.TensorSpec(shape=(), dtype=tf.int16) for c in categorical_columns}, # behaviors categorical columns
            'input_topics': tf.TensorSpec(shape=(window,N_TOPICS+1), dtype=tf.int32), # history topics sequence
            'input_category': tf.TensorSpec(shape=(window, 1), dtype=tf.int32), # history category sequence
            'input_subcategory': tf.TensorSpec(shape=(window, N_SUBCATEGORY+1), dtype=tf.int32), # history subcategory sequence
            'input_weekday': tf.TensorSpec(shape=(window, 1), dtype=tf.int32), # history weekday sequence
            'input_hour_group': tf.TensorSpec(shape=(window, 1), dtype=tf.int32), # history hour_group sequence
            'input_sentiment_label': tf.TensorSpec(shape=(window, 1), dtype=tf.int32), # history sentiment_label sequence
            'input_numerical': tf.TensorSpec(shape=(window, 3), dtype=tf.float32), # history (premium, read_time, scroll_percentage) sequence
        },
        tf.TensorSpec(shape=(), dtype=tf.float32), # target
    )
    
    training_dataset = tf.data.Dataset.from_generator(
        lambda : build_sequences_cls_iterator(history_seq_train, behaviors, window=window, numerical_columns=numerical_columns,
                                              categorical_columns=categorical_columns),
        output_signature=output_signature
    ).shuffle(buffer_size=65536).batch(128)
    
    validation_dataset = tf.data.Dataset.from_generator(
        lambda: build_sequences_cls_iterator(history_seq_val, behaviors_val, window=window, numerical_columns=numerical_columns,
                                             categorical_columns=categorical_columns),
        output_signature=output_signature
    ).batch(256, drop_remainder=True)

    model = AttentionHistoryClassificationModel(
        categorical_features=categorical_columns,
        numerical_features=numerical_columns,
        vocabulary_sizes=vocabulary_sizes,
        seq_embedding_dims={
            'input_topics': (N_TOPICS + 1, 20, True),
            'input_subcategory': (N_SUBCATEGORY + 1, 20, True),
            'input_category': (N_CATEGORY + 1, 20, False),
            'input_weekday': (N_WEEKDAY, 3, False),
            'input_hour_group': (N_HOUR_GROUP, 3, False),
            'input_sentiment_label': (N_SENTIMENT_LABEL + 1, 2, False)
        },
        seq_numerical_features=['scroll_percentage', 'read_time', 'premium'],
        window_size=window,
        embedding_dim=128,
        l1_lambda=1e-4,
        l2_lambda=1e-4,
        dropout_rate=0.2,
        query_n_layers=1,
        query_start_units=256,
        query_units_decay=1,
        query_activation='swish',
        dense_n_layers=4,
        dense_start_units=384,
        dense_units_decay=2,
        dense_activation='swish',
    )
    
    logging.info(f'Starting training and saving checkpoints to {checkpoint_dir}')
    
    model.fit(
        train_dataset=training_dataset,
        validation_data=validation_dataset,
        batch_size=None,
        epochs=25,
        early_stopping_rounds=5,
        loss=tfk.losses.BinaryCrossentropy(),
        optimizer=tfk.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=5.0),
        metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc')],
        lr_scheduler=get_simple_decay_scheduler(0.1, start_epoch=2),
        early_stopping_monitor='val_auc',
        early_stopping_mode='max',
        save_checkpoints=True,
        checkpoint_dir=checkpoint_dir
    )
    
    logging.info(f'Saving the model to {save_dir}')
    model.save(save_dir)
