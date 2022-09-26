import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from BreastCancerNet import BreastCancerNet

def prepare_dataloader():
    base = '/home/skerres/Studium/Cloud_Computing/FederatedLearning/BreastCancer/archive/'
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

    random.shuffle(data)
    data = data[:40000]
    images = torch.zeros((len(data), 3, 50, 50))
    labels = torch.zeros((len(data)))
    convert_tensor = transforms.ToTensor()
    for i, image_path in enumerate(data):
        label = int(image_path[-5])
        img = Image.open(image_path)
        img_resized = img.resize((50,50))
        img_normalized = np.array(img_resized) / 256
        img = convert_tensor(img_normalized)
        images[i] = img
        labels[i] = label

    dataset = torch.utils.data.TensorDataset(images, labels)
    
    batch_size = 256
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(images) * 0.7), int(len(images) * 0.3)])
    trainloader = DataLoader(train_dataset, batch_size, shuffle = True)
    testloader = DataLoader(test_dataset, batch_size, shuffle = True)
    return trainloader, testloader


def train(epoch, net, optimizer, criterion, trainloader):
    net.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(trainloader):
        ## Initialize the model, classify the images using the model, compute the loss between the real labels and predicted labels, backpropagate the loss and update the optimizer
        net.zero_grad()
        outputs = net(images)
        # outputs = outputs.long()
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        ## display the loss & epoch, every 10 batchs for instance.
        # print("Loss:"+ str(loss.item()) + ", Epoch:" + str(epoch) + ", Batch number "+ str(batch_idx))
    return total_loss

def test(model, criterion, testloader):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    true_positives, true_negatives, false_negatives, false_positives = 0.0, 0.0, 0.0, 0.0

    for batch_idx, (images, labels) in enumerate(testloader):
        #images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        labels = labels.view(-1, 1)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        outputs = outputs > 0.5
        confusion_vector = outputs / labels
        true_positives += torch.sum(confusion_vector == 1).item()
        false_positives += torch.sum(confusion_vector == float('inf')).item()
        true_negatives += torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives += torch.sum(confusion_vector == 0).item()
        # print("Positive Detection Accuracy: " + str(true_positives * 100/ (true_positives + false_negatives)) + "%")
        # print("Negative Detection Accuracy: " + str(true_negatives * 100 / (false_positives + true_negatives)) + "%")
        correct += torch.sum(torch.eq(outputs, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy*100, loss, true_positives, false_positives, true_negatives, false_negatives

def plot_confusion_matrix(precision_positive, recall_positive, precision_negative, recall_negative, train_loss, test_loss):
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 4))
    axs[0].plot(list(precision_positive.values()), label = "Precision")
    axs[0].plot(list(recall_positive.values()), label = "Recall")
    axs[0].legend()
    axs[0].set_xlabel("Epoch")
    axs[0].set_title("Images showing cancer cells")
    axs[0].set_ylabel("Percentage of accuracy and recall")
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axs[1].plot(list(precision_negative.values()), label = "P   recision")
    axs[1].plot(list(recall_negative.values()), label = "Recall")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Percentage of accuracy and recall")
    axs[1].set_title("Images not showing cancer cells")
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axs[1].legend()
    axs[2].plot(list(train_loss.values()), label = "Train loss")
    axs[2].plot(list(test_loss.values()), label = "Test loss")
    axs[2].set_title("Loss of training and test")
    axs[2].legend()
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Loss")

    plt.setp(axs, ylim=(0, 110))
    plt.savefig("centralized_model.png")
    # plt.show(block = True)


if __name__ == "__main__":
    trainloader, testloader = prepare_dataloader()
    print("Finished Data Preparation")
    net = BreastCancerNet()
    # history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 40, verbose = 2, batch_size = 256)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters())
    precision_positive, recall_positive, precision_negative, recall_negative = {}, {}, {}, {}
    train_loss, test_loss = {}, {}
    for epoch in range(30):
        accuracy, test_loss[epoch], true_positives, false_positives, true_negatives, false_negatives = test(net, criterion, testloader)
        train_loss[epoch] = train(epoch, net, optimizer, criterion, trainloader)
        if true_positives+false_positives == 0:
            precision_positive[epoch] = 0
        else:
            precision_positive[epoch] = true_positives * 100 / (true_positives + false_positives)
        if true_negatives + false_negatives == 0:
            precision_negative[epoch] = 0
        else:
            precision_negative[epoch] = true_negatives * 100 / (true_negatives + false_negatives)

        recall_positive[epoch] = true_positives * 100 / (true_positives + false_negatives)
        recall_negative[epoch] = true_negatives * 100 / (true_negatives + false_positives)
        print(f'Epoch: {epoch}',
                f'Accuracy: {accuracy}%',
                f'Loss: {test_loss[epoch]}',
                f'Precision class 1: {precision_positive[epoch]}%',
                f'Recall class 1 : {recall_positive[epoch]}%')

    plot_confusion_matrix(precision_positive, recall_positive, precision_negative, recall_negative, train_loss, test_loss)
