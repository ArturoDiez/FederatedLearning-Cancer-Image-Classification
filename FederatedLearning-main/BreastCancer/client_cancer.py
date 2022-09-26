import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torch
import copy
import sys
import re
import os
from google.cloud import storage
from BreastCancerNet import BreastCancerNet
import time
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn
from PIL import Image
import random



clientTest = {}
clientDict = {}
## Prepare Dataset
def prepare_dataset(client_id):
    storage_client = storage.Client()
    client_bucket_name = "client_bucket_ot6_426127"
    client_bucket = storage_client.get_bucket(client_bucket_name)
    datasets = {}
    for data_type in ["client_" + str(client_id), "test"]:
        image_path = os.path.join("client_data", data_type, data_type + "_images.pt")
        label_path = os.path.join("client_data", data_type, data_type + "_labels.pt")

        local_image_path = data_type + "_images.pt"
        blob_images = client_bucket.blob(image_path)
        # blob_images.download_to_filename(local_image_path)
        images = torch.load(local_image_path)

        local_labels_path = data_type + "_labels.pt"
        blob_labels = client_bucket.blob(label_path)
        # blob_labels.download_to_filename(local_labels_path)
        labels = torch.load(local_labels_path)

        datasets[data_type] = torch.utils.data.TensorDataset(images, labels)
    return datasets["client_" + str(client_id)], datasets["test"]


def train(epoch, net, optimizer, criterion, trainloader):
    net.train()
    for batch_idx, (images, labels) in enumerate(trainloader):
        ## Initialize the model, classify the images using the model, compute the loss between the real labels and predicted labels, backpropagate the loss and update the optimizer
        net.zero_grad()
        outputs = net(images)
        # outputs = outputs.long()
        labels = labels.view(-1, 1)
        # print("outputs: " + str(outputs))
        # print("labels: " + str(labels))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        ## display the loss & epoch, every 10 batchs for instance.
        # print("Loss:"+ str(loss.item()) + ", Epoch:" + str(epoch) + ", Batch number "+ str(batch_idx))

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


def get_initial_epoch(directory_name, storage_client, client_bucket_name):
    # Charge the day, it is the name of the current directory
    bucket = storage_client.bucket(client_bucket_name)

    filepaths = []
    filenames = []
    filenumbers = []

    # get all file names in given directory
    for blob in storage_client.list_blobs(bucket, prefix='mnist_weights/' + directory_name):
        filepaths.append(blob.name)

    # get epoch numbers in directory
    for path in filepaths:
        subpaths = path.split("/")
        filename = subpaths[-1]
        blob = bucket.blob(path)
        #fileExists = blob.exists()
        if "epoch" in filename:
            filenames.append(filename)
            blob.download_to_filename(filename)
            epochnumber = re.findall('epoch_(\d+)', filename)
            print(epochnumber)
            if len(epochnumber) != 1:
                epochvalue = int(epochnumber[0] + epochnumber[1])
            else:
                epochvalue = int(epochnumber[0])
            #print(epochvalue)
            filenumbers.append(epochvalue)

    if filenames != []:
        epoch = max(filenumbers)
    else:
        epoch = 0
    return epoch

def run_epoch(net, trainloader, testloader, epoch, storage_client):
    ## instantiate an optimizer for you model, as well as a criterion/loss function - Adam
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())

    # Evaluating the model before any training steps is a good practice
    accuracy, loss, true_positives, false_positives, true_negatives, false_negatives = test(net, criterion, testloader)
    positive_accuracy = true_positives * 100 / (true_positives + false_negatives)
    negative_accuracy = true_negatives * 100 / (false_positives + true_negatives)
    print(f'Epoch: {epoch}',
            f'Accuracy: {accuracy}%',
            f'Loss: {loss}',
            f'Positive Accuracy: {positive_accuracy}%',
            f'Negative Accuracy: {negative_accuracy}%')

    train(epoch, net, optimizer, criterion, trainloader)

def save_weights(net, storage_client, client_id, epoch, directory_name):

    client_filename = 'client_' + str(client_id) + '_epoch_' + str(epoch) + '.pt'
    torch.save(net.state_dict(), client_filename)

    bucketUp = storage_client.get_bucket("server_bucket_ot6_313078")
    client_filepath = 'mnist_weights/' + directory_name +'/'+ client_filename
    blobUp = bucketUp.blob(client_filepath)
    # @Aoife: Here you can insert the pub functionality for the client. 
    blobUp.upload_from_filename(client_filename)

def main(directory_name, client_id):
    """
        directory_name: string, name of the folder in the google cloud bucket in which the weights of the clients 
        are stored and from which the aggregated weights are loaded. Is a subfolder of mnist_weights/
        client_id: int, identifier of the client. 
    """

    learning_rate = 0.001
    batch_size = 256
    total_epochs = 100
    storage_client = storage.Client()
    client_bucket_name = "client_bucket_ot6_426127"
    #download mnist dataset and store in the training ditionary clientDict and the test dictioniary clientTest
    train_dataset, test_dataset = prepare_dataset(client_id)

    ## create two dataloaders (train & test) to load data into batches
    trainloader = DataLoader(train_dataset, batch_size, shuffle = True)
    testloader = DataLoader(test_dataset, batch_size, shuffle = True)
    # the initial epoch is zero when the target folders don't contain any weights yet
    epoch = get_initial_epoch(directory_name, storage_client, client_bucket_name)
    while epoch < total_epochs:
        # load the state dict 
        if epoch > 0:
            net.load_state_dict(torch.load("aggregated_epoch_" + str(epoch)))
        epoch += 1
        # run one training and testing step
        run_epoch(net, trainloader, testloader, epoch, storage_client)
        # upload updated state dict of NN to cloud
        save_weights(net, storage_client, client_id, epoch, directory_name)
        client_bucket = storage_client.bucket(client_bucket_name)
        filename = "aggregated_epoch_" + str(epoch)
        path = os.path.join("mnist_weights", directory_name, "aggregated_epoch_" + str(epoch) + ".pt")
        # wait for aggregated weight file which will be created by the server
        while True:
            # @Aoife: Here you can insert the sub functionality for the client. 
            # Here should probably a subscriber wait for a message that the server uploaded the aggregated weights to the client bucket
            stats = storage.Blob(bucket=client_bucket, name=path).exists(storage_client)
            print("waiting for file: " + str(path) + " exists?: " + str(stats), flush=True)
            if stats:
                time.sleep(1.0) #wait one extra second to improve chance that upload of file is finished
                break
            time.sleep(1.0)
        blob = client_bucket.blob(path)
        blob.download_to_filename(filename)


if __name__ == "__main__":
    # the credentials file must be in the same folder from which the client is started
    credentials_path = 'federatedlearninginsa-0caa7695b46b.json'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    directory_name = sys.argv[1]
    client_id = int(sys.argv[2])
    net = BreastCancerNet()
    main(directory_name, client_id)
