import tensorflow as tf


def get_simple_decay_scheduler(scheduling_rate):
    
    def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        else:
            return float(lr * tf.math.exp(-scheduling_rate))
        
    return tf.keras.callbacks.LearningRateScheduler(scheduler)