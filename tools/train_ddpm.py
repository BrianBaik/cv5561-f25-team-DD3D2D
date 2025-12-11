
print(">>> TRAIN_DDPM.PY WAS EXECUTED <<<")

import os
import sys

# Add project root (parent of "tools") to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



import torch
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import matplotlib.pyplot as plt   


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(">>> FILE LOADED <<<")

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Optional: limit number of batches per epoch (for fast debugging)
    max_batches = train_config.get('max_batches_per_epoch', None)
    
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    
    mnist = MnistDataset('train', im_path=dataset_config['im_path'])
    mnist_loader = DataLoader(
        mnist,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=0      # single-process dataloader (more stable on macOS)
    )
    print("MNIST dataset size =", len(mnist))

    model = Unet(model_config).to(device)
    model.train()
    
    # Create output directory
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # Noise error history  
    noise_error_history = []
    epoch_error_history = []

    # Load checkpoint if exists
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_path):
        print('Loading checkpoint...')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch_idx in range(num_epochs):
        print(">>> STARTING TRAIN LOOP <<<")
        losses = []
        for batch_idx, im in enumerate(tqdm(mnist_loader)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            optimizer.zero_grad()
            im = im.float().to(device)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Random timestep
            t = torch.randint(
                0,
                diffusion_config['num_timesteps'],
                (im.shape[0],)
            ).to(device)

            # Add noise
            noisy_im = scheduler.add_noise(im, noise, t)

            # Predict noise
            noise_pred = model(noisy_im, t)

            # Loss = MSE(true_noise, predicted_noise)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())

            # record average noise error per batch
            noise_error_history.append(loss.item())

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch_idx + 1} | Loss: {np.mean(losses):.4f}")
        print(">>> FINISHED TRAIN LOOP <<<")
        torch.save(model.state_dict(), ckpt_path)

    print("Done Training.")

    plt.figure(figsize=(8, 4))
    plt.plot(noise_error_history, label="Noise Prediction MSE", linewidth=1)
    plt.xlabel("Training Iteration")
    plt.ylabel("MSE")
    plt.title("Noise Prediction Error During Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(train_config['task_name'], "noise_error_curve.png"))
    plt.show()
    plt.close()

    print(
        "Saved noise error curve →",
        os.path.join(train_config['task_name'], "noise_error_curve.png")
    )

        # NEW: per-epoch MSE plot
        plt.figure(figsize=(6, 4))
        epochs = range(1, len(epoch_loss_history) + 1)
        plt.plot(epochs, epoch_loss_history, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Noise Prediction MSE per Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(train_config['task_name'], "epoch_mse_curve.png"))
        plt.show()
        plt.close()

        print("Saved epoch MSE curve →",
            os.path.join(train_config['task_name'], "epoch_mse_curve.png"))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path', default='config/default.yaml', type=str) 
    args = parser.parse_args() 
    train(args)