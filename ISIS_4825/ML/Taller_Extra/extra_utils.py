import numpy as np
import os
from . import general as gen
from tensorflow import keras

def get_vol_slice(vol, lab, idx):
    vol_slice = vol[..., idx]
    lab_slice = lab[..., idx]
    vol_slice = np.rot90(vol_slice, 1)
    lab_slice = np.rot90(lab_slice, 1)

    vol_slice = gen.scale(vol_slice, 0, 255)
    return vol_slice, lab_slice

def save_to_dir(X, index_name, y=None, data_folder="/content/train_data"):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    X_data_folder = os.path.join(data_folder, "data")
    if not os.path.exists(X_data_folder):
        os.makedirs(X_data_folder)
    
    np.save(os.path.join(X_data_folder, f"X_{index_name}"), X)

    if y is not None:
        y_data_folder = os.path.join(data_folder, "labels")
        if not os.path.exists(y_data_folder):
            os.makedirs(y_data_folder)
        np.save(os.path.join(y_data_folder, f"y_{index_name}"), y)

def get_labeled_image(img, label, num_classes, as_categorical=True):
    if as_categorical:
        aux_label = keras.utils.to_categorical(label, num_classes=num_classes, 
                                           dtype="uint8")
    else:
        aux_label = label.copy()

    labeled_image = np.zeros_like(aux_label, dtype=img.dtype)

    for i in range(labeled_image.shape[-1]):
        labeled_image[..., i] = img * aux_label[..., 0]
    
    labeled_image[..., 1:] += aux_label[..., 1:] * img.max()
    
    return labeled_image

def get_slice(X, y, num_classes=3, bg_thresh=0.98):
    y = keras.utils.to_categorical(y, num_classes, "uint8")
    s = y.shape
    bg_ratio = y[..., 0].sum() / (s[0] * s[1])
    if bg_ratio < bg_thresh:
        y = y[..., 1: ]
        X = gen.scale(X, 0, 255)
        return X, y.astype("uint8")

    return None