import os
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from google.cloud import storage
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from BreastCancerNet import BreastCancerNet

base = '/home/skerres/Studium/Cloud_Computing/FederatedLearning/BreastCancer/archive/'
target = '/home/skerres/Studium/Cloud_Computing/FederatedLearning/BreastCancer/client_data/'
os.chdir('/home/skerres/Studium/Cloud_Computing/FederatedLearning/BreastCancer/archive/')
ids = os.listdir(base)
data = []
for id in ids:
    try:
        files1 = os.listdir(base + id + '/1/')
        files0 = os.listdir(base + id + '/0/')
        for x in files1:
            data.append(base + id + '/1/' + x)
        for x in files0:
            data.append(base + id + '/0/' + x)
    except:
        FileNotFoundError
os.chdir(target)

data = data[:20000]

number_clients = 5
images_per_client = len(data) / number_clients
client_id = 0
images, labels = {}, {}
images = torch.zeros((len(data), 3, 50, 50))
labels = torch.zeros((len(data)))
convert_tensor = transforms.ToTensor()

for i, image_path in enumerate(data):
    if i % images_per_client == 0 and i != 0:
        client_filepath = "client_" + str(client_id) + "_images.pt"
        torch.save(images, client_filepath)
        client_filepath = "client_" + str(client_id) + "_labels.pt"
        torch.save(labels, client_filepath)
        print("Stored data")
        print(client_filepath)
        images = torch.zeros((len(data), 3, 50, 50))
        labels = torch.zeros((len(data)))
        client_id += 1
    label = int(image_path[-5])
    img = Image.open(image_path)
    img_resized = img.resize((50,50))
    img_normalized = np.array(img_resized) / 256
    img = convert_tensor(img_normalized)
    images[i] = img
    labels[i] = label