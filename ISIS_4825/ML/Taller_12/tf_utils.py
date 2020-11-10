from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, weights_dir, patience=10, rate=0.5):
        super(CustomCallback, self).__init__()
        self.weights_dir = weights_dir
        self.rate = rate
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.best_loss = np.Inf
        self.best_recall = 0
        self.wait = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        current_recall = logs.get("val_Recall")
        if (current_loss < self.best_loss or 
            current_recall > self.best_recall):
            self.wait = 0
            self.best_loss = current_loss
            self.best_recall = current_recall
            self.model.save_weights(self.weights_dir)
            self.best_weights = self.model.get_weights()
            print("\nBest Weights Saved!!")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEpoch {epoch}, Reducing Learning Rate")
                lr = K.get_value(self.model.optimizer.lr)
                new_lr = lr * self.rate
                K.set_value(self.model.optimizer.lr, new_lr)
                print(f"\nLearning Rate Reduced: {new_lr}")
                self.model.set_weights(self.best_weights)
                print("\nBest Weights Loaded!!")

def DenseBlock(units):
    return keras.Sequential([
        keras.layers.Dense(units, activation="relu"),
        keras.layers.BatchNormalization()
    ])