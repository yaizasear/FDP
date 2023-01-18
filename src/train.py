import torch
from torch import nn


def train(model: nn.Module, train_dataloader, criterion, optimizer, num_classes, epoch, device) -> float:
    model.train()
    total_loss = 0.
    num_batches = sum(1 for batch in train_dataloader)
    for step, batch in enumerate(train_dataloader):
        data, targets = batch[0], batch[1]
        targets_input = targets[:,:-1].type(torch.LongTensor).to(device) # shifted right
        targets_expected = targets[:,1:].type(torch.LongTensor).to(device)
        output = model(data, targets_input)
        loss = criterion(output, targets_expected)
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_batches

def validate(model: nn.Module, val_dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.
    num_batches = sum(1 for batch in val_dataloader)
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            data, targets = batch[0], batch[1]
            targets_input = targets[:, :-1].type(torch.LongTensor).to(device)
            targets_expected = targets[:,1:].type(torch.LongTensor).to(device)
            output = model(data, targets_input)
            total_loss += criterion(output, targets_expected).item()
    return total_loss / num_batches