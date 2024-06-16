import polars as pl
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import numpy as np
import logging
import random
import os
from datetime import datetime

from polimi.utils.tf_models.utils.build_sequences import build_history_seq, build_sequences_seq_iterator, N_CATEGORY, N_SENTIMENT_LABEL, N_SUBCATEGORY, N_TOPICS, N_HOUR_GROUP, N_WEEKDAY
from polimi.utils.tf_models import TemporalHistorySequenceModel, TemporalHistoryClassificationModel

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(seed)


if __name__ == '__main__':
    history = pl.read_parquet('/home/ubuntu/dataset/ebnerd_small/train/history.parquet')
    articles = pl.read_parquet('/home/ubuntu/dataset/ebnerd_small/articles.parquet')

    history_seq = build_history_seq(history, articles)
    window = 20
    stride = 10

    # Define the output signature for the tuple of dictionaries
    output_signature = (
        {
            "input_topics": tf.TensorSpec(shape=(window,N_TOPICS+1), dtype=tf.int32),
            "input_category": tf.TensorSpec(shape=(window,1), dtype=tf.int32),
            'input_subcategory': tf.TensorSpec(shape=(window, N_SUBCATEGORY+1), dtype=tf.int32), # subcategory
            'input_weekday': tf.TensorSpec(shape=(window, 1), dtype=tf.int32), # weekday
            'input_hour_group': tf.TensorSpec(shape=(window, 1), dtype=tf.int32), # hour_group
            'input_sentiment_label': tf.TensorSpec(shape=(window, 1), dtype=tf.int32), # sentiment_label
            'input_numerical': tf.TensorSpec(shape=(window, 3), dtype=tf.float32), # (premium, read_time, scroll_percentage)
        },
        {
            "output_topics": tf.TensorSpec(shape=(N_TOPICS + 1,), dtype=tf.int32),
            "output_category": tf.TensorSpec(shape=(N_CATEGORY + 1, ), dtype=tf.int32),
            'output_subcategory': tf.TensorSpec(shape=(N_SUBCATEGORY + 1,), dtype=tf.int32), # subcategory
            'output_sentiment_label': tf.TensorSpec(shape=(N_SENTIMENT_LABEL + 1,), dtype=tf.int32), # sentiment_label
        }
    )

    # Create the dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        lambda: build_sequences_seq_iterator(history_seq, window=window, stride=stride, target_telescope_type='random_same_day'),
        output_signature=output_signature
    )
    # first shuffle, then batch, otherwise the buffer size will be the number of batches, not the number of samples
    dataset = dataset.shuffle(buffer_size=65536).batch(128)

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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f'/home/ubuntu/experiments/rnn_seq_{timestamp}'
    checkpoint_dir = f'{save_dir}/checkpoints'
    os.makedirs(checkpoint_dir)
    model.fit(
        train_dataset=dataset,
        batch_size=256,
        epochs=5,
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
