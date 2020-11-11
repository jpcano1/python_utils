import copy

from torch.nn import functional as F
import torch

from tqdm.auto import tqdm

def jaccard(y_pred, y_true, dim=(2, 3), eps=1e-5):
    inter = torch.sum(y_true * y_pred, dim=dim)
    union = torch.sum(y_pred, dim=dim) + torch.sum(y_true, dim=dim)
    union -= inter
    IoU = ((inter + eps) / (union + eps)).mean()
    loss = 1 - IoU
    return loss, IoU

def loss_func(y_pred, y_true, metric):
    loss, acc = metric(y_pred, y_true)
    return loss, acc

def batch_loss(criterion, y_pred, y_true, metric, opt=None):
    loss, acc = criterion(y_pred, y_true, metric)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), acc.item()

def epoch_loss(model, criterion, metric, dataloader, device, sanity_check=False, opt=None):
    epoch_loss = 0.
    epoch_acc = 0.
    len_data = len(dataloader)

    for X_batch, y_batch in tqdm(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)

        b_loss, b_acc = batch_loss(criterion, y_pred, y_batch, metric, opt)
        epoch_loss += b_loss

        if b_acc is not None:
            epoch_acc += b_acc
            
        if sanity_check:
            break

    loss = epoch_loss / float(len_data)
    acc = epoch_acc / float(len_data)
    return loss, acc

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def train(model, epochs, criterion, opt, train_dl, val_dl, 
          sanity_check, lr_scheduler, weights_dir, device, metric=jaccard, **kwargs):
    loss_history = {
        "train": [],
        "val": []
    }

    acc_history = {
        "train": [],
        "val": []
    }

    best_model = copy.deepcopy(model.state_dict())
    best_loss = kwargs.get("best_loss") or float("inf")
    best_acc = kwargs.get("best_acc") or 0

    for _ in tqdm(range(epochs)):
        current_lr = get_lr(opt)

        model.train()
        train_loss, train_acc = epoch_loss(model, criterion, metric, train_dl, 
                                           device, sanity_check, opt)
        loss_history["train"].append(train_loss)
        acc_history["train"].append(train_acc)

        model.eval()

        with torch.no_grad():
            val_loss, val_acc = epoch_loss(model, criterion, metric, val_dl, 
                                           device, sanity_check)
        
        loss_history["val"].append(val_loss)
        acc_history["val"].append(val_acc)

        if val_loss < best_loss or val_acc > best_acc:
            best_loss = val_loss
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            
            torch.save(model.state_dict(), weights_dir)
            print("Copied best model weights!")
        
        lr_scheduler.step(val_loss)

        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model)

        print(f"Train Loss: {train_loss:.6f}, Accuracy: {100 * train_acc:2f}")
        print(f"Val loss: {val_loss:.6f}, Accuracy: {100 * val_acc:.2f}")
        print("-"*50)

        if sanity_check:
            break

    model.load_state_dict(best_model)
    return model, loss_history, acc_history