import os
import sys
import time
import yaml
import random
import numpy as np
from torch.optim import Adam
from transformer import Transformer
from lightning.fabric import Fabric
from torch.nn import CrossEntropyLoss
from datasets import ProteinSequenceDataset
from torch.optim.lr_scheduler import StepLR
from train import train, validate, MLMpretrain
from torch import set_float32_matmul_precision
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, SequentialSampler


def main(config):
    set_float32_matmul_precision('medium')

    # Init Wandb and Fabric
    wandb_logger = WandbLogger(
        project= config["wandb_project"],
        name= config["wandb_name"],
        config={
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"]})

    fabric = Fabric(accelerator="gpu", 
                    devices=config["devices"],
                    precision="16-mixed",
                    strategy="deepspeed",
                    loggers=wandb_logger)

    fabric.launch()

    # Define task
    pretrain = config["pretrain"]
    model_name = "pretraining" if pretrain else "finetuning"
    random_id = str(random.randint(100000, 999999))
    directory_path = os.path.join(config["out_dir"], model_name + "-" + random_id)
    if fabric.global_rank == 0:
        os.makedirs(directory_path)

    # Datasets and dataloaders
    upsampling = not pretrain
    train_path = config["pretrain_data"] if pretrain else config["train_data"]
    file_format = config["pretrain_file_format"] if pretrain else config["train_file_format"]
    train_dataset = ProteinSequenceDataset(train_path, file_format, upsampling=upsampling, masking=pretrain)
    num_tokens = train_dataset.num_tokens 
    pad_length = train_dataset.pad_length
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler, 
                                  drop_last=True, num_workers=config["num_workers"], pin_memory=True)
    if not pretrain:
        val_dataset = ProteinSequenceDataset(config["val_data"], file_format, pad_length, upsampling=upsampling, masking=pretrain)
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], sampler=val_sampler, num_workers=30, pin_memory=True)
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    else:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # Show metrics
    fabric.print("Protein Language Model (", random_id, ")\n", "-" * 89)
    fabric.print("Training dataset: ", len(train_dataset), "sequences")
    fabric.print("Training batches: ", len(train_dataloader) * config["devices"]) 
    if not pretrain:
        fabric.print("Validation dataset: ", len(val_dataset), "sequences")
        fabric.print("Validation batches: ", len(val_dataloader))
    fabric.print("Batch size: ", config["batch_size"])
    fabric.print("Sequence length: ", pad_length, "\n")
    fabric.print("Number of tokens: ", num_tokens)
    fabric.print("Number of workers: ", config["num_workers"], "\n")

    # Model
    model = Transformer(num_tokens)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), config["lr"])
    scheduler = StepLR(optimizer, config["lr_step_size"], config["gamma"])
    model, optimizer = fabric.setup(model, optimizer)

    # Load pretrained weights (if any)
    if not pretrain:
        state = {"model": model, "optimizer": optimizer}
        fabric.load(config["weights"], state)

    # Save initial embedding weights
    initial_weights = model.embedding.weight
    save_path = os.path.join(directory_path, "initial_embedding_weights.tsv")
    np.savetxt(save_path, initial_weights.cpu().detach().numpy(), delimiter='\t', fmt='%.6f')

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    fabric.print(f"Total number of parameters: {total_params}", "\n", "-" * 89)

    # Train/validation loop
    print(f"Training on {fabric.device} with {fabric.strategy} strategy")
    for epoch in range(1, config["epochs"] + 1):
        epoch_start_time = time.time()

        if pretrain:
            train_loss, train_acc = MLMpretrain(fabric, model, criterion, optimizer, train_dataloader)
            fabric.log_dict({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
        else:
            train_loss = train(fabric, model, criterion, optimizer, train_dataloader)
            val_loss = validate(model, criterion, val_dataloader)
            fabric.log_dict({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        elapsed = time.time() - epoch_start_time

        if pretrain:
            fabric.print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | lr: {lr:2.6f} |'
                f' train loss: {train_loss:5.6f} | train acc: {train_acc:5.6f} |')
        else:
            fabric.print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | lr: {lr:2.6f} |'
                f' train loss: {train_loss:5.6f} | val loss: {val_loss:5.6f} |')
        
        # Save model and embedding weights
        save_path = os.path.join(directory_path, "epoch_" + str(epoch) + ".ckpt")
        state = {"model": model, "optimizer": optimizer}
        fabric.save(save_path, state)
        embedding_weights = model.embedding.weight
        weights_path = os.path.join(directory_path, "embedding_weights_" + str(epoch) + ".tsv")
        np.savetxt(weights_path, embedding_weights.cpu().detach().numpy(), delimiter='\t', fmt='%.6f')


if __name__ == '__main__':
    with open('./configs/plm_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)
