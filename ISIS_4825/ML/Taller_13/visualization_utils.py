import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

from . import general as gen


def get_labeled_image(img, label, outline_color=(1, 0, 0), color=(1, 0, 0), mode="outer"):
    """
    Function to get the labeled image
    :param img: The image
    :param label: The mask label
    :param outline_color: The color of outline
    :param color: The color of fill
    :return: The mask and the image merged
    """
    assert mode in ['thick', 'inner', 'outer', 'subpixel']
    img_mask = mark_boundaries(img, label, outline_color=outline_color, color=color, mode=mode)
    return img_mask


def predict(model, device, dataset, class_: str = "kidney", random_state=1234, **kwargs):
    """
    Method to make predictions from a model
    :param model: The model that makes the prediction
    :param device: The hardware accelerator device
    :param dataset: The dataset to make the predictions
    :param class_: The class of the predictions
    :param random_state: The random seed
    """
    if class_ == "kidney":
        channel = 0
    elif class_ == "tumor":
        channel = 1
    else:
        raise Exception("No es la clase esperada")

    # Random seed
    np.random.seed(random_state)
    # Take the random sample
    random_sample = np.random.choice(len(dataset), 3)

    # Create the figure to plot on
    plt.figure(figsize=(12, 12))
    for i in range(len(random_sample)):
        rnd_idx = random_sample[i]
        # Take the image and the label
        X, y_true = dataset[rnd_idx]
        y_true = y_true[channel]
        # Extend dims and allocate on device
        X_t = X.unsqueeze(0).to(device)
        X = X.squeeze(0)
        # Predict
        y_pred = model(X_t)
        y_pred = y_pred.squeeze(0)
        y_pred = y_pred[channel].cpu().detach().numpy() > 0.5

        # Plot the results versus the originals
        plt.subplot(3, 4, 1 + i * 4)
        gen.imshow(X, color=False, cmap="bone", title="Image")

        plt.subplot(3, 4, 2 + i * 4)
        gen.imshow(y_pred, color=False, title=f"Predicted {class_.title()}")

        plt.subplot(3, 4, 3 + i * 4)
        gen.imshow(get_labeled_image(X, y_pred, **kwargs), title="Boundary")

        plt.subplot(3, 4, 4 + i * 4)
        gen.imshow(y_true, color=False, title="True Label")


def rugosity(image):
    if image.max() != 0:
        var = image.var()
        max = image.max() ** 2
        return 1 - (1 / (1 + var / max))
    else:
        return 0
