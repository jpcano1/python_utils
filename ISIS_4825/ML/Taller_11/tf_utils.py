from tensorflow import keras
from tensorflow.keras import backend as K

class ModelSaveCallback(keras.callbacks.Callback):
    def __init__(self, filename):
        super(ModelSaveCallback, self).__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        model_filename=self.filename.format(epoch)
        keras.models.save_model(self.model, model_filename)
        print(f"\nModel saved in {model_filename}")

class LRHistory(keras.callbacks.Callback):
    def __init_(self, model):
        super(LRHistory, self).__init__()
        self.model = model

    def on_epoch_begin(self, epoch, logs={}):
        print(f"Learning Rate: {K.get_value(self.model.optimizer.lr)}")