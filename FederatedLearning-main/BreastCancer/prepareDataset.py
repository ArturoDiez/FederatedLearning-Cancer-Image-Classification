from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
import os
import zipfile
from PIL import Image
from google.cloud import storage

# Data of 1/4 of the images (see drive)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\GCP\_federatedlearninginsa-65abb6c7d304.json"

def main():
    trainDict = {}
    subsetDict = {}
    clientSetDict = {}
    sizeDatasets = {}
    sizeDatasetsClients = {}
    subsetsNestDict = {}
    clientDict = {}

    storage_client = storage.Client()
    bucket = storage_client.get_bucket("server_bucket_ot6_313078")
    bucketUp = storage_client.get_bucket("client_bucket_ot6_426127")
    base = 'dataset'
    data = []

    filename = 'subset_breast_cancer_data.zip'
    blob = bucketUp.blob(filename)
    #blob.download_to_filename(filename)

    #with zipfile.ZipFile(filename, 'r') as zip_ref:
    #    zip_ref.extractall(base)


    ids = os.listdir(base)
    for id in ids:
        try:
            files1 = os.listdir(base + '/'+ id + '/1/')
            files0 = os.listdir(base + '/'+ id + '/0/')
            for x in files1:
                data.append(base + '/'+ id + '/1/' + x)
            for x in files0:
                data.append(base + '/'+ id + '/0/' + x)
        except:
            FileNotFoundError


    data = data[:20000]
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
    train_size = int(len(images) * 0.75)
    test_size = int(len(images) * 0.25)
    print(len(images))
    print(len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    torch.save(test_dataset, 'test_dataset')

    alpha = 1

    tdataset_filepath = 'client_dataset/alpha_' + str(alpha) + '/test_dataset'
    blob = bucketUp.blob(tdataset_filepath)

    blob.upload_from_filename('test_dataset')
    print(test_dataset)

    nusers = 5
    ndata = 2

    diri = np.random.dirichlet([alpha]*nusers , ndata).transpose()

    labels_train = []
    for images, labels in train_dataset:
        labels_train.append(int(labels))
    labels = labels_train
    numbersRg = [0, 1]
    for i in numbersRg:
      trainDict['train{0}'.format(i)] = []

    for j in numbersRg:
      for i in range(len(labels)):
        tensor = str(j)
        if str(labels[i]) == tensor:
          trainDict['train{0}'.format(j)].append(i)

    for i in numbersRg:
       subsetDict['subset{0}'.format(i)] = Subset(train_dataset, trainDict['train{0}'.format(i)])

    numberClients = list(range(nusers))
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
        print('client' + str(j) + ' received ' + str(sizeDatasetsClients['n' + str(i) + '_client' + str(j)]) + ' images from class' + str(i))

    for j in numberClients:
        dataset_filename = 'client{0}data'.format(j)

        torch.save(clientDict['client{0}set'.format(j)], dataset_filename)

        dataset_filepath = 'client_dataset/alpha_' + str(alpha) + '/' + dataset_filename

        blobUp = bucketUp.blob(dataset_filepath)


        blobUp.upload_from_filename(dataset_filename)

        print(dataset_filename)

if __name__ == "__main__":
    main()
