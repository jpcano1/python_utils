import numpy as np
import os

def get_slice(vol, lab, idx):
    vol_slice = vol[..., idx]
    lab_slice = lab[..., idx]
    vol_slice = np.rot90(vol_slice, 1)
    lab_slice = np.rot90(lab_slice, 1)

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