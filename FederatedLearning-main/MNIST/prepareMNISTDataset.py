from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\GCP\_federatedlearninginsa-65abb6c7d304.json"
storage_client = storage.Client()

def main():
    trainDict = {}
    subsetDict = {}
    clientSetDict = {}
    sizeDatasets = {}
    sizeDatasetsClients = {}
    subsetsNestDict = {}
    clientDict = {}

    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='data',train = True, transform = transform, download = True)
    test_dataset = datasets.MNIST(root='data',train= False, transform = transform)
    torch.save(test_dataset, 'test_dataset')

    bucketUp = storage_client.get_bucket("client_bucket_ot6_426127")
    tdataset_filepath = 'mnist_weights/dataset/test_dataset'
    blob = bucket.blob(tdataset_filepath)

    blob.upload_from_filename('test_dataset')

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
      clientDict['client{0}set'.format(j)] = []

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
        clientDict['client{0}set'.format(j)].extend(subsetsNestDict['client' + str(j) + '_set' + str(i)])

    for j in numberClients:
        dataset_filename = 'client{0}data'.format(j)
        torch.save(clientDict['client{0}set'.format(j)], dataset_filename)

        dataset_filepath = 'mnist_weights/dataset/' + dataset_filename
        blobUp = bucket.blob(dataset_filepath)

        blobUp.upload_from_filename(dataset_filename)
        print(dataset_filename)

if __name__ == "__main__":
    main()
