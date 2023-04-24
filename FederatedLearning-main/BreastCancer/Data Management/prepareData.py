from torch.utils.data import TensorDataset
from torchvision import transforms
import numpy as np
import torch
import os
from PIL import Image
from google.cloud import storage
import random
import math


def main():
    datasets_dict = {}

    storage_client = storage.Client()
    # bucket = storage_client.get_bucket("server_bucket_ot6_313078")
    bucket_up = storage_client.get_bucket("client_bucket_ot6_426127")
    base = 'dataset'
    data = []
    total_blobs = len(list(storage_client.list_blobs('server_bucket_ot6_313078', prefix=base)))
    for i, blob in enumerate(storage_client.list_blobs('server_bucket_ot6_313078', prefix=base)):
        names = blob.name
        subpaths = names.split("/")
        filename = subpaths[-1]
        data.append(filename)
        blob.download_to_filename('data/' + filename)
        if i % 500 == 0:
            print("Download progress: " + str(i) + "/" + str(total_blobs))
 
    # random.shuffle(data)

    test_quantity = int(len(data) * 0.25)

    test = data[:test_quantity]
    datasets_dict['test'] = test
    train = data[test_quantity:]

    alpha = 10
    n_users = 5
    n_data = 2

    distribution = np.random.dirichlet([alpha] * n_users, n_data).transpose()

    for numData in list(range(n_data)):
        train_sub = []
        for path in train:
            if int(path[-5]) == numData:
                train_sub.append(path)

        length = int(len(train_sub))

        initial = 0

        for user in list(range(n_users)):
            amount = math.floor(length * int(distribution[user][numData]))

            final = initial + amount

            slices = train_sub[initial:final]

            datasets_dict['client_{0}'.format(user)] = []
            datasets_dict['client_{0}'.format(user)].extend(slices)

            initial = final

    convert_tensor = transforms.ToTensor()
    for data_type in datasets_dict.keys():

        images = torch.zeros((len(datasets_dict[data_type]), 3, 50, 50))
        labels = torch.zeros((len(datasets_dict[data_type])))

        for j, image_path in enumerate(datasets_dict[data_type]):
            label = int(image_path[-5])
            # print(label)
            img = Image.open('data/' + image_path)
            img_resized = img.resize((50, 50))
            img_normalized = np.array(img_resized) / 256
            img = convert_tensor(img_normalized)
            images[j] = img
            labels[j] = label

        datasets_dict[data_type + '_images'] = TensorDataset(images)
        datasets_dict[data_type + '_labels'] = TensorDataset(labels)

        for h in ['_images', '_labels']:
            files_name = data_type + h
            torch.save(datasets_dict[files_name], files_name)

            dataset_filepath = os.path.join("client_data_leo", data_type, files_name)
            blob_up = bucket_up.blob(dataset_filepath)

            blob_up.upload_from_filename(files_name)


if __name__ == "__main__":
    credentials_path = 'federatedlearninginsa-0caa7695b46b.json'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    main()
