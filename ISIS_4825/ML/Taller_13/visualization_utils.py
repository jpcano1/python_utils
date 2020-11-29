from . import general as gen
from . import visualization_utils as vis
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_tensor
from skimage.segmentation import mark_boundaries

def get_labeled_image(img, label, outline_color=(1, 0, 0), 
                        color=(1, 0, 0)):
    """
    Function to get the labeled image
    :param img: The image
    :param label: The mask label
    :param outline_color: The color of outline
    :param color: The color of fill
    :return: The mask and the image merged
    """
    img_mask = mark_boundaries(img, label, outline_color=outline_color, 
                               color=color, mode="thick")
    return img_mask

def predict(model, device, dataset, class_: str="kidney", random_state=None):
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

    # Set the random seed
    if random_state:
        np.random.seed(random_state)

    # Take the random sample
    random_sample = np.random.choice(len(dataset), 3)

    # Create the figure to plot on
    plt.figure(figsize=(12, 12))
    for i in range(len(random_sample)):
        # Take the image and the label
        X, y_true = dataset[i]
        y_true = y_true[channel]
        # Extend dims and allocate on device
        X_t = X.unsqueeze(0).to(device)
        X = X.squeeze(0)
        # Predict
        y_pred = model(X_t)
        y_pred = y_pred.squeeze(0)
        y_pred = y_pred[channel].cpu().detach().numpy() > .5

        # Plot the results versus the originals
        plt.subplot(3, 4, 1 + i*4)
        gen.imshow(X, color=False, cmap="bone", title="Image")

        plt.subplot(3, 4, 2 + i*4)
        gen.imshow(y_pred, color=False, title=f"Predicted {class_.title()}")

        plt.subplot(3, 4, 3 + i*4)
        gen.imshow(vis.mark_boundaries(X, y_pred), title="Boundary")

        plt.subplot(3, 4, 4 + i*4)
        gen.imshow(y_true, color=False, title="True Label")