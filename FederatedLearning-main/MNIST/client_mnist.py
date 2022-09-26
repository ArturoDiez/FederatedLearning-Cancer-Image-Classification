import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torch
import copy
import sys
import re
import os
from google.cloud import storage
from MNIST_model import Net
import time
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn



clientTest = {}
clientDict = {}
## Prepare Dataset
def prepareData():
    print("Preparing Dataset")
    trainDict = {}
    subsetDict = {}
    clientSetDict = {}
    sizeDatasets = {}
    sizeDatasetsClients = {}
    subsetsNestDict = {}

    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='data',train = True, transform = transform, download = True)
    test_dataset = datasets.MNIST(root='data',train= False, transform = transform)
    clientTest['test'] = test_dataset

    alpha = 5
    nusers = 5
    ndata = 10

    diri = np.random.dirichlet([alpha]*nusers , ndata).transpose()

    labels = train_dataset.targets
    numbersRg = list(range(0,9))
    for i in numbersRg:
      trainDict['train{0}'.format(i)] = []

    for j in numbersRg:
      for i in range(len(labels)):
        tensor = "tensor(" + str(j) + ")"
        if str(labels[i]) == tensor:
          trainDict['train{0}'.format(j)].append(i)

    for i in numbersRg:
       subsetDict['subset{0}'.format(i)] = Subset(train_dataset, trainDict['train{0}'.format(i)])

    labels = train_dataset.targets
    numberClients = list(range(nusers))
    #print(numberClients)
    for j in numberClients:
      clientDict['client{0}set'.format(j+1)] = []

    for i in numbersRg:
      sizeDatasets['n{0}'.format(i)] = len(subsetDict['subset{0}'.format(i)])
      for j in numberClients:
        if j == 0:
          newRg = 0
          lastRg = 0
        sizeDatasetsClients['n' + str(i) + '_client' + str(j)] = int(diri[j,i] * sizeDatasets['n{0}'.format(i)])
        newRg += sizeDatasetsClients['n' + str(i) + '_client' + str(j)]
        diricSize = range(lastRg, newRg)
        #print(diricSize)
        lastRg = newRg + 1
        subsetsNestDict['client' + str(j) + '_set' + str(i)] = Subset(subsetDict['subset{0}'.format(i)], diricSize)
        clientDict['client{0}set'.format(j+1)].extend(subsetsNestDict['client' + str(j) + '_set' + str(i)])
    print("Finished Preparation")


def train(epoch, net, optimizer, criterion, trainloader):
    net.train()
    for batch_idx, (images, labels) in enumerate(trainloader):
        ## Initialize the model, classify the images using the model, compute the loss between the real labels and predicted labels, backpropagate the loss and update the optimizer
        net.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        ## display the loss & epoch, every 10 batchs for instance.
        # print("Loss:"+ str(loss.item()) + ", Epoch:" + str(epoch) + ", Batch number "+ str(batch_idx))
def test(model, criterion, testloader):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    for batch_idx, (images, labels) in enumerate(testloader):
        #images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy*100, loss

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # Evaluating the model before any training steps is a good practice
    accuracy, loss = test(net, criterion, testloader)
    print('Before training :',f'Accuracy: {accuracy}', f'Loss: {loss}' )

    train(epoch, net, optimizer, criterion, trainloader)
    accuracy, loss = test(net, criterion, testloader)
    print(f'Epoch: {epoch}', f'Accuracy: {accuracy}', f'Loss: {loss}' )

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
    prepareData()

    ## create two dataloaders (train & test) to load data into batches
    trainloader = DataLoader(clientDict['client1set'], batch_size, shuffle = True)
    testloader = DataLoader(clientTest['test'], batch_size, shuffle = True)
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
    net = Net()
    main(directory_name, client_id)
