import os
import sys
import yaml
import time
import torch
import random
import matplotlib.pyplot as plt
from gan import Generator, Discriminator
from datasets import ProteinDataset, ProteinDataLoader, get_d_model
from blast import makeblastdb
from train import train, validate


def main(config):
    print("ProteinGAN\n")
    print("-" * 105)

    # Generate experiment id and output dir
    random_id = str(random.randint(100000, 999999))
    os.makedirs(os.path.join(config["out_dir"], "proteinGAN_" + random_id))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train/validation datasets
    train_file = config["train_data"]
    val_file = config["val_data"]
    train_dataset = ProteinDataset(train_file, config)
    val_dataset = ProteinDataset(val_file, config)
    
    # Blast database
    makeblastdb(config, random_id)
    
    # Model dimension
    max_d_model = config["max_d_model"]
    d_model = get_d_model(train_dataset, val_dataset, max_d_model)
    print("Dimension of the model (d_model): ", d_model)
    
    # Train/validation data loaders
    batch_size = config["batch_size"]
    train_dataloader = ProteinDataLoader(train_dataset, batch_size, d_model, device)
    val_dataloader = ProteinDataLoader(val_dataset, batch_size, d_model, device)
    print("Train dataset (without upsampling): ", len(train_dataset), "sequences")
    print("Train dataset (with upsampling): ", sum(1 for batch in train_dataloader) * batch_size, "sequences")
    print("Validation dataset: ", len(val_dataset), "sequences")
    print("Steps/epoch: ", sum(1 for batch in train_dataloader))
    print("-" * 105)
    
    # Model
    generator = Generator(config, d_model, device).to(device)
    discriminator = Discriminator(config, d_model, device).to(device)
    lr_g = config["lr_g"]
    lr_d = config["lr_d"]
    lr_step_size = config["lr_step_size"]
    lr_gamma = config["lr_gamma"]
    betas = (config["beta1"], config["beta2"])
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, lr_step_size, gamma=lr_gamma)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, lr_step_size, gamma=lr_gamma)
    
    # Training/validation loop
    print("Training...")
    epochs = config["epochs"]
    noise_size = config["z_dim"]
    train_losses_g, train_losses_d, val_losses_g, val_losses_d = [], [], [], []
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss_g, train_loss_d = train(generator, discriminator, train_dataloader, optimizer_g, optimizer_d, noise_size, epoch, random_id, config, device)
        val_loss_g, val_loss_d = validate(generator, discriminator, val_dataloader, noise_size, batch_size, device)
        train_losses_g.append(train_loss_g)
        train_losses_d.append(train_loss_d)
        val_losses_g.append(val_loss_g)
        val_losses_d.append(val_loss_d)
        lr_g = scheduler_g.get_last_lr()[0]
        lr_d = scheduler_d.get_last_lr()[0]
        scheduler_g.step()
        scheduler_d.step()

        elapsed = time.time() - epoch_start_time
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | lr g: {lr_g:2.6f} | lr d: {lr_d:2.6f} |'
              f' train loss g: {train_loss_g:5.6f} | train loss d: {train_loss_d:5.6f} |'
              f' val loss g: {val_loss_g:5.6f} | val loss d: {val_loss_d:5.6f} |')
    
        if epoch > 4:
            # Plot train/validation loss curves
            epoch_vec = [i for i in range(1, epoch + 1)]

            plot_g_path = os.path.join(config["out_dir"], "proteinGAN_" + random_id, "loss_g_gan_" + str(epoch) + ".png")
            plt.figure(figsize=(15,7.5))
            plt.title("Generator - Training and Validation Loss")
            plt.plot(epoch_vec, train_losses_g, label="Training")
            plt.plot(epoch_vec, val_losses_g, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(plot_g_path)
            plt.close()

            plot_d_path = os.path.join(config["out_dir"], "proteinGAN_" + random_id, "loss_d_gan_" + str(epoch) + ".png")
            plt.figure(figsize=(15,7.5))
            plt.title("Discriminator - Training and Validation Loss")
            plt.plot(epoch_vec, train_losses_g, label="Training")
            plt.plot(epoch_vec, val_losses_g, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(plot_d_path)
            plt.close()

            plot_gd_path = os.path.join(config["out_dir"], "proteinGAN_" + random_id, "loss_gd_gan_" + str(epoch) + ".png")
            plt.figure(figsize=(15,7.5))
            plt.title("Generator and Discriminator - Training Loss")
            plt.plot(epoch_vec, train_losses_g, label="Generator")
            plt.plot(epoch_vec, train_losses_d, label="Discriminator")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(plot_gd_path)
            plt.close()

        # Save model
        if epoch % 10 == 0:
            PATH = os.path.join(config["out_dir"], "proteinGAN_" + random_id, "pgan_epoch_" + str(epoch) + ".pt")
            torch.save({
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "d_model": d_model}, PATH)

    # Best model
    print('=' * 105)
    print(f' | End of training | lr g: {lr_g:2.2f} | lr d: {lr_d:2.2f} |'
            f' val loss g: {val_loss_g:5.2f} | val loss d: {val_loss_d:5.2f} |')
    print('=' * 105)
    
    # Save model
    PATH = os.path.join(config["out_dir"], "proteinGAN_" + random_id, "proteingan_end.pt")
    torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'd_model': d_model}, PATH)

if __name__ == '__main__':
    with open('./configs/pgan_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)



