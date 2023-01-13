import time
import math
import torch
from torch import nn


def train(model: nn.Module, train_dataloader, criterion, optimizer, scheduler, num_classes, epoch, device) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 150
    num_batches = sum(1 for batch in train_dataloader)
    start_time = time.time()

    for i, batch in enumerate(train_dataloader):
        data, targets = batch[0], batch[1]
        targets_input = targets[:, :-1].type(torch.LongTensor).to(device) # shifted right
        targets_expected = targets[:,1:].type(torch.LongTensor).to(device)
        output = model(data, targets_input)
        loss = criterion(output, targets_expected)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/i {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def validate(model: nn.Module, val_dataloader, criterion, d_model, device) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    num_batches = sum(1 for batch in val_dataloader)
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            data, targets = batch[0], batch[1]
            targets_input = targets[:, :-1].type(torch.LongTensor).to(device)
            targets_expected = targets[:,1:].type(torch.LongTensor).to(device)
            output = model(data, targets_input)
            total_loss += d_model * criterion(output, targets_expected).item()
    return total_loss / ((num_batches * 64) - 1)