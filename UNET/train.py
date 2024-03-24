import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time

def train(model, data_loader, optimizer, loss_function, device):
    epoch_loss = 0.0

    model.train()
    for x, y in data_loader: 
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward() # backpropagation
        optimizer.step() # update the weights
        epoch_loss += loss.item() # loss.item() คือ ค่า loss ของแต่ละ batch

    epoch_loss = epoch_loss/len(data_loader) # หารด้วยจำนวน batch ที่เทรน
    return epoch_loss

def evaluate(model, data_loader, loss_function, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_function(y_pred, y)
            epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(data_loader)
    return epoch_loss


if __name__ == '__main__':
    # seeding
    seeding(42)

    # create a directory
    create_dir('files')

    # Load dataset
    x_train = sorted(glob('new_data/train/image/*'))
    y_train = sorted(glob('new_data/train/mask/*'))

    x_valid = sorted(glob('new_data/test/image/*'))
    y_valid = sorted(glob('new_data/test/mask/*'))
    print(f"Number of train images: {len(x_train)}")
    print(f"Number of test images: {len(x_valid)}")

    # Hyperparameters
    batch_size = 2
    lr = 1e-4
    num_epochs = 50
    checkpoint_path = 'files/checkpoint.pth'  # Path to save the model

    # Dataset and DataLoader
    train_dataset = DriveDataset(x_train, y_train)
    valid_dataset = DriveDataset(x_valid, y_valid)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Device setup
    device = torch.device('cuda')
    print(f'Using {device} device.')

    # Model, Optimizer, and Loss
    model = build_unet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    loss_function = DiceBCELoss()

    # Training
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_function, device)
        valid_loss = evaluate(model, validation_loader, loss_function, device)

        # save the model
        if valid_loss < best_valid_loss:
            data_str =  f'Valid loss improved from {best_valid_loss:.4f} to {valid_loss:.4f}'
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, Time: {epoch_mins}m {epoch_secs}s')
