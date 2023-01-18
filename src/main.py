import os
import sys
import yaml
import copy
import time
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
from transformer import Transformer
from protein import ProteinDataset, ProteinDataLoader, get_d_model
from train import train, validate


def main():
    print("TRANSFORMER\n")
    print("-" * 89)
    
    # Parameters
    device = torch.device("cuda")
    with open(sys.path[0] + '/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    batch_size = config["batch_size"]
    
    # Train/validation datasets
    train_file = config["train_data"]
    val_file = config["val_data"]
    train_dataset = ProteinDataset(train_file, config)
    val_dataset = ProteinDataset(val_file, config)
    
    # Model dimension
    max_d_model = config["d_model"]
    d_model = get_d_model(train_dataset, val_dataset, max_d_model)
    print("Dimension of the model (d_model): ", d_model)
    
    # Train/validation data loaders
    train_dataloader = ProteinDataLoader(train_dataset, batch_size, d_model, device)
    val_dataloader = ProteinDataLoader(val_dataset, batch_size, d_model, device)
    print("Train dataset (without upsampling): ", len(train_dataset), "sequences")
    print("Train dataset (with upsampling): ", sum(1 for batch in train_dataloader) * 64, "sequences")
    print("Validation dataset: ", len(val_dataset), "sequences")
    print("-" * 89, "\n")
    
    # Model
    num_classes = config["num_classes"]
    model = Transformer(device, num_classes, d_model).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = config["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # Train/validation loop
    print("Training...")
    epochs = config["epochs"]
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(model, train_dataloader, criterion, optimizer, num_classes, epoch, device)
        val_loss = validate(model, val_dataloader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr = scheduler.get_last_lr()[0]
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
        
        scheduler.step()
        
        elapsed = time.time() - epoch_start_time
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | lr: {lr:2.2f} |'
              f' train loss: {train_loss:5.2f} | val loss: {val_loss:5.2f} |')
        
        if epoch == 50:
            break
    
    # Plot train/validation loss curve
    plot_path = os.path.join(config["data_dir"], "loss_transformer.png")
    epoch_vec = [i for i in range(1, 50 + 1)] # epochs not 50
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(epoch_vec, train_losses, label="Training")
    plt.plot(epoch_vec, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(plot_path)

    # Best model
    test_loss = validate(best_model, val_dataloader, criterion, device) # test_dataloader if test data
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | lr: {lr:2.2f} | best valid loss {test_loss:5.2f} |')
    print('=' * 89)

    # Save model
    PATH = os.path.join(config["data_dir"], "transformer.pt")
    torch.save({ 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, PATH)

if __name__ == '__main__':
    main()