import torch

def jaccard(y_pred, y_true, dim=(2, 3), eps=1e-5,
            reduction="mean", *args, **kwargs):
    """
    Intersection over Union metric
    :param y_pred: The predictions of the model
    :param y_true: The true labels of the data
    :param dim: The axis where we calculate the operations
    :param eps: The tolerance
    :return: The loss calculated and the IoU metric
    """
    assert reduction in ["sum", "mean"], "Reduction non valid"
    # Intersection
    inter = torch.sum(y_true * y_pred, dim=dim)
    # Union
    union = torch.sum(y_pred, dim=dim) + torch.sum(y_true, dim=dim)
    union -= inter

    IoU = (inter + eps) / (union + eps)
    # The whole metric
    if reduction == "mean":
        IoU = IoU.mean()
    else:
        IoU = IoU.sum()
    loss = 1 - IoU
    return loss, IoU

def dice(y_pred, y_true, dim=(2, 3), eps=1e-5, 
         reduction="mean", *args, **kwargs):
    """
    Dice Similarity metric, Sørensen index
    :param y_pred: The predictions of the model
    :param y_true: The true labels of the data
    :param dim: The axis where we calculate the operations
    :param eps: The tolerance
    :return: The loss calculated and the dice metric
    """
    assert reduction in ["sum", "mean"], "Reduction non valid"
    # Intersection
    num = 2 * torch.sum(y_pred * y_true, dim=dim) + eps
    # Pseudo union
    den = torch.sum(y_pred + y_true, dim=dim) + eps

    # The whole metric
    dice = num / den
    if reduction == "mean":
        dice = dice.mean()
    else:
        dice = dice.sum()
    loss = 1 - dice
    return loss, dice