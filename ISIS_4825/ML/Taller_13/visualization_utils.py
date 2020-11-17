from . import general as gen
from . import visualization_utils as vis
import numpy as np
import matplotlib.pyplot as plt

import cv2
from torchvision.transforms.functional import to_tensor
from skimage.segmentation import mark_boundaries

def get_labeled_image(img, label, outline_color=(1, 0, 0), 
                        color=(1, 0, 0)):
    """

    :param img:
    :type img:
    :param label:
    :type label:
    :param outline_color:
    :type outline_color:
    :param color:
    :type color:
    :return:
    :rtype:
    """
    img_mask = mark_boundaries(img, label, outline_color=outline_color, 
                               color=color, mode="thick")
    return img_mask

def predict(model, random_sample, device, data_dir, labels_dir, class_="kidney"):
    """

    :param model:
    :type model:
    :param random_sample:
    :type random_sample:
    :param device:
    :type device:
    :param data_dir:
    :type data_dir:
    :param labels_dir:
    :type labels_dir:
    :param class_:
    :type class_:
    :return:
    :rtype:
    """
    if class_ == "kidney":
        channel = 0
    elif class_ == "tumor":
        channel = 1
    else:
        raise Exception("No es la clase esperada")

    plt.figure(figsize=(12, 12))
    for i in range(len(random_sample)):
        rnd_idx = random_sample[i]
        X = np.load(data_dir[rnd_idx])
        y_true = np.load(labels_dir[rnd_idx])[..., channel]
        X = cv2.resize(X, (128, 128), cv2.INTER_NEAREST)
        X_t = to_tensor(X).unsqueeze(0).to(device)
        y_pred = model(X_t)
        y_pred = y_pred.squeeze(0)
        y_pred = y_pred[channel].cpu().detach().numpy() > .5
        
        plt.subplot(3, 4, 1 + i*4)
        gen.imshow(X, color=False, cmap="bone", title="Image")

        plt.subplot(3, 4, 2 + i*4)
        gen.imshow(y_pred, color=False, title="Predicted Tumor")

        plt.subplot(3, 4, 3 + i*4)
        gen.imshow(vis.mark_boundaries(X, y_pred), title="Boundary")

        plt.subplot(3, 4, 4 + i*4)
        gen.imshow(y_true, color=False, title="True Label")