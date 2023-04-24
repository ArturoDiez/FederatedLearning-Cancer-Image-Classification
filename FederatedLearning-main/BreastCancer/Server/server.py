import sys
from google.cloud import storage
from BreastCancerNet import BreastCancerNet
import torch
import copy 
import time
import os
from client_cancer import get_initial_epoch

def aggregrate_models(net_dict, base_net): 
    update_state = {}
    first = True
    for k in net_dict.keys():
        # print(update_state["conv1.weight"][0])
        for key in net_dict[k].state_dict().keys():
            # print("Updating weights for key: " + str(key) + " and client: " + str(k))
            if first:
                update_state[key] = net_dict[k].state_dict()[key]
            else:
                update_state[key] += net_dict[k].state_dict()[key]
        first = False
    for key in update_state:
         update_state[key] = update_state[key] / len(net_dict.keys())
    base_net.load_state_dict(copy.deepcopy(update_state))
    return base_net

def download_weight_files(blob, bucket, client_weight_paths):
    # download client weight files
    filenames = []
    for path in client_weight_paths:
        subpaths = path.split("/")
        filename = subpaths[-1]
        blob = bucket.blob(path)
        filenames.append(filename)
        blob.download_to_filename(filename)
    return filenames

def load_client_dict(client_weights_paths, filenames):
    # load all client models
    client_dict = {}
    number_clients = len(client_weights_paths)

    for i in range(number_clients):
        model = BreastCancerNet()
        model.load_state_dict(torch.load(filenames[i]))
        client_dict[i] = copy.deepcopy(model)
    return client_dict

def main():
    if len(sys.argv) != 2:
        print("Provide exactly one argument which is the folder name of the client weights, e.g. \"python3 server.py 2021_12_20_17_11_13\"")
        return
    max_wait_time = 10
    max_clients = 3
    max_epoch = 100
    storage_client = storage.Client()
    server_bucket_name = "server_bucket_ot6_313078"
    server_bucket = storage_client.get_bucket(server_bucket_name)
    # get all files in the subdirectory of mnist_weights which is specified on the commandline
    directory_name = sys.argv[1]

    # select all client weight files with a matching epoch
    epoch = get_initial_epoch(directory_name, storage_client, server_bucket_name)
    while epoch < max_epoch:
        first = True
        while True: 
            client_weights_paths = []
            filepaths = []
            epoch_substring = "epoch_" + str(epoch)
            for blob in storage_client.list_blobs(server_bucket_name, prefix='mnist_weights/' + directory_name):
                filepaths.append(blob.name)
            for filepath in filepaths:
                if epoch_substring in filepath:
                    client_weights_paths.append(filepath)

            clients_finished = len(client_weights_paths)
            if clients_finished > 0 and first:
                time_start = time.time()
                first = False
            # when one client is finished a waiting process is started. Wait until either a number of clients given by the variable clients_finished are finished
            # or until the time limit max_wait_time passed.
            # @Aoife Here you can add the sub functionality for the server
            if clients_finished > 0:
                print("Clients finished:" + str(clients_finished)
                        + "/" + str(max_clients) + ", time passed: "
                        + str(int(time.time() - time_start)) + "s/"
                        + str(max_wait_time) + "s", flush=True)
            else:
                print("Clients finished:" + str(clients_finished)
                        + "/" + str(max_clients), flush=True)
            if clients_finished > 0 and (clients_finished >= max_clients or int(time.time() - time_start) >= max_wait_time):
                break
            time.sleep(1.0)
        print("client_weights_paths: " + str(client_weights_paths))
        filenames = download_weight_files(blob, server_bucket, client_weights_paths)
        client_dict = load_client_dict(client_weights_paths, filenames)

        # aggregate them into the new server model 
        server_model = BreastCancerNet()
        server_model = aggregrate_models(client_dict, server_model)
        server_filename = "aggregated_epoch_" + str(epoch) + ".pt"
        torch.save(server_model.state_dict(), server_filename)

        # upload weights of the server to the client bucket
        client_bucket = storage_client.get_bucket("client_bucket_ot6_426127")
        server_filepath = os.path.join("mnist_weights", directory_name, server_filename)
        blob = client_bucket.blob(server_filepath)
        # @Aoife here you can add the pub functionality for the server
        blob.upload_from_filename(server_filename)
        print("Uploaded weights to path: " + server_filepath + " for epoch " + str(epoch))
        epoch = epoch + 1

if __name__ == "__main__":
    credentials_path = 'federatedlearninginsa-0caa7695b46b.json'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    main()