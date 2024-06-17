import tensorflow as tf


def get_simple_decay_scheduler(scheduling_rate, start_epoch=5):
    
    def scheduler(epoch, lr):
        if epoch < start_epoch:
            return lr
        else:
            return float(lr * tf.math.exp(-scheduling_rate))
        
    return tf.keras.callbacks.LearningRateScheduler(scheduler)