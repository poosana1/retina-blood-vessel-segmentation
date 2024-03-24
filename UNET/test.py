import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    # Ground truth
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5 
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    # Prediction
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    # metrics
    jaccard = jaccard_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return [jaccard, f1, recall, precision, accuracy]

def mask_parse(mask): 
    mask = np.expand_dims(mask, axis=-1) #(512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1) #(512, 512, 3)
    return mask

if __name__ == '__main__':
    # seeding
    seeding(42)

    # create a directory
    create_dir('results')

    # load dataset
    x_test = sorted(glob('new_data/test/image/*'))
    y_test = sorted(glob('new_data/test/mask/*'))

    # Hyperparameters
    Height = 512
    Width = 512
    size = (Height, Width)
    checkpoint_path = 'files/checkpoint.pth'  # Path to save the model

    # load the checkpoint
    device = torch.device('cuda')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(x_test, y_test)), total=len(x_test)):
        # Extract name
        name = os.path.split(x)[-1].split('.')[0]
        # print(name)

        # break

        # reading image
        image = cv2.imread(x, cv2.IMREAD_COLOR) #(512, 512, 3)
        # image = cv2.resize(image, size)
        x = np.transpose(image, (2,0,1)) #(3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0) #(1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        # reading mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE) #(512, 512)
        # mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0) #(1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0) #(1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            # prediction and calculate the metrics
            start_time = time.time()
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, y_pred)
            metrics_score = list(map(add, metrics_score, score))
            y_pred = y_pred[0].cpu().numpy() #(1, 512, 512)
            y_pred = np.squeeze(y_pred, axis=0) #(512, 512)
            y_pred = y_pred > 0.5
            y_pred = np.array(y_pred, dtype=np.uint8)

        # save the results
        original_mask = mask_parse(mask)
        y_pred = mask_parse(y_pred)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate([image, line, original_mask, line, y_pred * 255], axis=1)
        cv2.imwrite(f'results/{name}.png', cat_images)

    jaccard_sc = metrics_score[0]/len(x_test)
    f1_sc = metrics_score[1]/len(x_test)
    recall_sc = metrics_score[2]/len(x_test)
    precision_sc = metrics_score[3]/len(x_test)
    accuracy_sc = metrics_score[4]/len(x_test)
    print(f'Jaccard: {jaccard_sc:.4f} - F1: {f1_sc:.4f} - Recall: {recall_sc:.4f} - Precision: {precision_sc:.4f} - Accuracy: {accuracy_sc:.4f}') 

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)