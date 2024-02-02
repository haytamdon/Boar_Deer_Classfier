import torch
from torch import nn
from tqdm import tqdm

def train_fn(loader, model, optimizer, loss_fn, device):
    """
    Training loop function

    Args:
        loader (DataLoader):    Training DataLoader
        model (Model):          Model to train
        optimizer (Optimizer):  Optimization algorithm
        loss_fn (nn.Module):    Loss function for the model training
        device (str):           Device on which to train 'cpu' or 'cuda'

    Returns:
        train_loss (float):     calculated loss for the training
        train_acc (float):      calculated dice score for the training
    """
    model.train()
    train_loss, train_acc = 0, 0
    for batch_idx, (data, targets) in enumerate(loader):

        data = data.to(device=device)
        targets = targets.to(device=device)
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predictions_class = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
        train_acc += (predictions_class == targets).sum().item()/len(predictions)

    train_loss = train_loss / len(loader)
    train_acc = train_acc / len(loader)
    return train_loss, train_acc

def val_fn(loader, model, device, loss_fn):
    """
    Validation Loop function

    Args:
        loader (DataLoader):    Test DataLoader
        model (Model):          Model to test
        device (str):           Device on which to train 'cpu' or 'cuda'
        loss_fn (nn.Module):    Loss function for model testing

    Returns:
        softdicescore (float):  calculated loss for the testing
        val_score (float):      calculated dice score for the testing
    """
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.inference_mode():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device=device)
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            test_pred_labels = outputs.argmax(dim=1)
            test_acc += ((test_pred_labels == targets).sum().item()/len(test_pred_labels))
    test_loss = test_loss / len(loader)
    test_acc = test_acc / len(loader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
        epochs: int = 5):

    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_fn(model=model,
                                        loader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device = device)
        test_loss, test_acc = val_fn(model=model,
                                    loader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device= device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

