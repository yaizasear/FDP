import torch
from tqdm import tqdm
from vocabulary import TOKENS


def train(fabric, model, criterion, optimizer, train_dataloader):
    model.train()
    total_loss = 0.
    for batch in tqdm(train_dataloader):
        trg = batch[:,:-1] 
        expected = batch[:,1:]
        logits = model(batch, trg)
        loss = criterion(logits, expected)
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(train_dataloader)

def validate(model, criterion, val_dataloader):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            trg = batch[:,:-1] 
            expected = batch[:,1:]
            logits = model(batch, trg)
            loss = criterion(logits, expected)
            total_loss += loss.item()
    return total_loss / len(val_dataloader)

def MLMpretrain(fabric, model, criterion, optimizer, train_dataloader):
    model.train()
    train_loss, train_acc, total_unmasked = 0, 0, 0
    for batch in tqdm(train_dataloader):
        src = batch[:,0]
        trg = batch[:,0,:-1] 
        mask = batch[:,1,1:] < TOKENS.index(b'MASK')
        expected = batch[:,1,1:][mask] 
        n = mask.float().sum().item() 
        total_unmasked += n
        logits = model(src, trg).permute(0, 2, 1)[mask]  
        loss = criterion(logits, expected)
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        predicted = torch.argmax(logits, 1)
        correct = torch.sum((expected == predicted).float()).item()
        loss_delta = n * (loss.item() - train_loss)
        train_loss += loss_delta/n
        acc_delta = correct - (n * train_acc)
        train_acc += acc_delta/n
    return train_loss, train_acc