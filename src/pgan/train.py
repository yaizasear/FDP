import torch
import numpy as np
from torch import nn
from datasets import ProteinDataLoader
from blast import runblast


def train(generator: nn.Module, discriminator: nn.Module, train_dataloader: ProteinDataLoader, 
            optimizer_g: torch.optim, optimizer_d: torch.optim,
            noise_size: int, epoch: int, jobid:str, config: dict, device: torch.device):
    generator.train()
    discriminator.train()
    g_losses, d_losses = 0., 0.
    batch_size = config["batch_size"]
    num_batches = sum(1 for batch in train_dataloader)
    step = (epoch - 1) * num_batches
    steps_for_blast = config["steps_for_blast"]
    relu = torch.nn.ReLU()

    for i, real_samples in enumerate(train_dataloader):
        # Optimize discriminator
        discriminator.zero_grad()
        predictions_d_real = discriminator(real_samples)
        loss_d_real = torch.mean(relu(1.0 - predictions_d_real)) 
        loss_d_real.backward()
        noise = torch.Tensor(np.random.normal(0, 0.5, (batch_size, noise_size))).to(device)
        fake_samples = generator(noise)
        predictions_d_fake = discriminator(fake_samples.detach())
        loss_d_fake = torch.mean(relu(1.0 + predictions_d_fake))
        loss_d_fake.backward(retain_graph=True) 
        loss_d = loss_d_real + loss_d_fake
        d_losses = d_losses + loss_d.item() 
        optimizer_d.step()

        # Optimize generator
        generator.zero_grad()
        noise = torch.Tensor(np.random.normal(0, 0.5, (batch_size, noise_size))).to(device)
        fake_samples = generator(noise)
        predictions_d_fake = discriminator(fake_samples)
        loss_g = torch.mean(0.0 - predictions_d_fake)
        # Run Blast
        if ((step + i + 1) % steps_for_blast == 0) and ((step + i) != 0):
            blast_score, avg_identity, max_identity = runblast(fake_samples.detach(), (step + i + 1), jobid, config)
            loss_g = loss_g + (abs(loss_g) * blast_score)
            print(f"    blast:  step {(step + i + 1):5d} | avg identity {avg_identity:5.2f} | max identity {max_identity:5.2f} ")
        loss_g.backward()
        g_losses = g_losses + loss_g.item()
        optimizer_g.step()

    return g_losses / num_batches, d_losses / num_batches


def validate(generator: nn.Module, discriminator: nn.Module, val_dataloader: ProteinDataLoader,
                noise_size: int, batch_size: int, device: torch.device):
    generator.eval()
    discriminator.eval()
    relu = torch.nn.ReLU()
    g_losses, d_losses = 0., 0.
    num_batches = sum(1 for batch in val_dataloader)
    with torch.no_grad():
        for real_samples in val_dataloader:
            # Evaluate discriminator
            predictions_d_real = discriminator(real_samples)
            loss_d_real = torch.mean(relu(1.0 - predictions_d_real))
            noise = torch.Tensor(np.random.normal(0, 0.5, (batch_size, noise_size))).to(device)
            fake_samples = generator(noise)
            predictions_d_fake = discriminator(fake_samples.detach())
            loss_d_fake = torch.mean(relu(1.0 + predictions_d_fake))
            loss_d = loss_d_real + loss_d_fake
            d_losses += loss_d.item()

            # Evaluate generator
            noise = torch.Tensor(np.random.normal(0, 0.5, (batch_size, noise_size))).to(device)
            fake_samples = generator(noise)
            predictions_d_fake = discriminator(fake_samples)
            loss_g = torch.mean(0.0 - predictions_d_fake)
            g_losses += loss_g.item()
    return g_losses / num_batches, d_losses / num_batches
