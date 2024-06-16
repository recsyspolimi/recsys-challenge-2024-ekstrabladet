import polars as pl
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import numpy as np
import logging
import random
import os
from datetime import datetime
import gc

from polimi.utils.tf_models.utils.build_sequences import build_history_seq, build_sequences_seq_iterator, N_CATEGORY, N_SENTIMENT_LABEL, N_SUBCATEGORY, N_TOPICS, N_HOUR_GROUP, N_WEEKDAY
from polimi.utils.tf_models import TemporalHistoryClassificationModel, TemporalHistorySequenceModel

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(seed)

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
import logging


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f'/home/ubuntu/experiments/rnn_seq_{timestamp}'
    os.makedirs(save_dir)
    
    log_path = os.path.join(save_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))
    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    dtype='ebnerd_testset'
    dsplit = 'test'
    logging.info(f'Reading dataset {dtype}')
    history_train = pl.read_parquet(f'/home/ubuntu/dataset/{dtype}/{dsplit}/history.parquet')
    # history_val = pl.read_parquet(f'/home/ubuntu/dataset/{dtype}/validation/history.parquet')
    articles = pl.read_parquet(f'/home/ubuntu/dataset/{dtype}/articles.parquet')
    
    #Sampling
    gap = 0.2
    history_train = history_train.filter(
        pl.col('article_id_fixed').list.len() >= pl.col('article_id_fixed').list.len().quantile(gap),
        pl.col('article_id_fixed').list.len() <= pl.col('article_id_fixed').list.len().quantile(1-gap), 
    )
    # history_val = history_val.filter(
    #     pl.col('article_id_fixed').list.len() >= pl.col('article_id_fixed').list.len().quantile(gap),
    #     pl.col('article_id_fixed').list.len() <= pl.col('article_id_fixed').list.len().quantile(1-gap), 
    # )
    logging.info(f'Sampled {len(history_train)} users out of {len(history_train)}...')
    
    window = 20
    stride = 10
    
    logging.info('Building history seq...')
    history_seq_train = build_history_seq(history_train, articles)
    # history_seq_val = build_history_seq(history_val, articles)
    logging.info('...done')
    
    del history_train, articles
    gc.collect()
    
    # Define the output signature for the tuple of dictionaries
    output_signature = (
        {
            "input_topics": tf.TensorSpec(shape=(window,N_TOPICS+1), dtype=tf.int16),
            "input_category": tf.TensorSpec(shape=(window,1), dtype=tf.int16),
            'input_subcategory': tf.TensorSpec(shape=(window, N_SUBCATEGORY+1), dtype=tf.int16), # subcategory
            'input_weekday': tf.TensorSpec(shape=(window, 1), dtype=tf.int16), # weekday
            'input_hour_group': tf.TensorSpec(shape=(window, 1), dtype=tf.int16), # hour_group
            'input_sentiment_label': tf.TensorSpec(shape=(window, 1), dtype=tf.int16), # sentiment_label
            'input_numerical': tf.TensorSpec(shape=(window, 3), dtype=tf.float32), # (premium, read_time, scroll_percentage)
        },
        {
            "output_topics": tf.TensorSpec(shape=(N_TOPICS + 1,), dtype=tf.int16),
            "output_category": tf.TensorSpec(shape=(N_CATEGORY + 1, ), dtype=tf.int16),
            'output_subcategory': tf.TensorSpec(shape=(N_SUBCATEGORY + 1,), dtype=tf.int16), # subcategory
            'output_sentiment_label': tf.TensorSpec(shape=(N_SENTIMENT_LABEL + 1,), dtype=tf.int16), # sentiment_label
        }
    )

    batch_size=128
    buffer_size=65536
    # Create the dataset from the generator
    dataset_train = tf.data.Dataset.from_generator(
        lambda: build_sequences_seq_iterator(history_seq_train, window=window, stride=stride, target_telescope_type='random_same_day'),
        output_signature=output_signature
    )
    # first shuffle, then batch, otherwise the buffer size will be the number of batches, not the number of samples
    dataset_train = dataset_train.shuffle(buffer_size=buffer_size).batch(batch_size)
    
    # # Create the dataset from the generator
    # dataset_val = tf.data.Dataset.from_generator(
    #     lambda: build_sequences_seq_iterator(history_seq_val, window=window, stride=stride, target_telescope_type='random_same_day'),
    #     output_signature=output_signature
    # )
    # # first shuffle, then batch, otherwise the buffer size will be the number of batches, not the number of samples
    # dataset_val = dataset_val.batch(batch_size)

    model = TemporalHistorySequenceModel(
        seq_embedding_dims={
            # adding one dim more to cover missings, where needed
            'input_topics': (N_TOPICS + 1, 10, True),
            'input_subcategory': (N_SUBCATEGORY + 1, 10, True),
            'input_category': (N_CATEGORY + 1, 10, False),
            'input_weekday': (N_WEEKDAY, 3, False),
            'input_hour_group': (N_HOUR_GROUP, 3, False),
            'input_sentiment_label': (N_SENTIMENT_LABEL + 1, 2, False)
        },
        seq_numerical_features=['scroll_percentage', 'read_time', 'premium'],
        n_recurrent_layers=1,
        recurrent_embedding_dim=64,
        l1_lambda=1e-4,
        l2_lambda=1e-4,
    )
    checkpoint_dir = f'{save_dir}/checkpoints'
    os.makedirs(checkpoint_dir)
    logging.info(f'Starting training and saving to {save_dir}')
    
    model.fit(
        train_dataset=dataset_train,
        # validation_data=dataset_val,
        early_stopping_rounds=2,
        batch_size=None,
        epochs=10,
        # target for (topics, subcategory, category)
        loss={
            'output_topics': tfk.losses.BinaryCrossentropy(), 
            'output_subcategory': tfk.losses.BinaryCrossentropy(), 
            'output_category': tfk.losses.CategoricalCrossentropy(),
            'output_sentiment_label': tfk.losses.CategoricalCrossentropy(),
        },
        loss_weights={
            'output_topics': 0.5, 
            'output_subcategory': 0.1, 
            'output_category': 0.3,
            'output_sentiment_label': 0.1,
        },
        optimizer=tfk.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-3, clipvalue=1.0),
        save_checkpoints=True,
        checkpoint_dir=checkpoint_dir
    )
    
    model.save(save_dir)
