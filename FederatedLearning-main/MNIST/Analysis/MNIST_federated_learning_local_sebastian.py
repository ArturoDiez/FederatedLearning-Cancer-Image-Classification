import torch
from torchvision import datasets, transforms
from IPython import display
from base64 import b64decode
from time import time
import copy 
from MNIST_model import Net
import numpy as np
import random
from datetime import datetime
import os




def get_dataset():  
## Prepare Dataset

    """ Returns train and test datasets for MNIST
        Data is normalized using (0.1307,), (0.3081,)
    """
    transforms_applied = transforms.Compose([
    transforms.ToTensor(),
   # transforms.Normalize((0.1307,), (0.3081,)),
     ])

    transforms_normalize = transforms.Normalize((0.1307,), (0.3081,))
    train_dataset = datasets.MNIST(
          "MNIST/processed/", train = True, download = True, transform = transforms_applied)
    test_dataset = datasets.MNIST(
        "MNIST/processed/", train = False, download = True, transform = transforms_applied)
            # "MNIST/processed/", train = False, download = True, transform = transforms_normalize)
    return train_dataset, test_dataset


def train(epoch, client_net, optimizer, criterion, trainloader_client):
    loss = 0.0
    client_net.train()
    for batch_idx, (images, labels) in enumerate(trainloader_client):
        #images, labels = images.to(device), labels.to(device)

        # Inference
        optimizer.zero_grad()
        outputs = client_net(images)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        ## Initialize the model
        ##  classify the images using the model
        ## compute the loss between the real labels and predicted labels
        ## backpropagate the loss
        ## update the optimizer
        ## display the loss & epoch, every 10 batchs for instance.
    return client_net
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

def main():
    print("Initializing")
    datasets_dirichlet_train_x = {}
    datasets_dirichlet_train_y = {}
    for i in range(10):
        datasets_dirichlet_train_x[i] = []
        datasets_dirichlet_train_y[i] = []



    train_dataset, test_dataset = get_dataset()
    alpha = 10
    for i, object in enumerate(iter(train_dataset)):
        base_probability = np.ones(10)
        base_probability[object[1]] = alpha
        prob = np.random.dirichlet(base_probability)
        dataset_id = np.random.choice(10, p=prob)
        datasets_dirichlet_train_x[dataset_id].append(object[0])
        datasets_dirichlet_train_y[dataset_id].append(object[1])


    dataloader_train = {}
    for i in range(10):
        tensor_x = torch.Tensor(len(datasets_dirichlet_train_x[i]), 28, 28)
        torch.cat(datasets_dirichlet_train_x[i], out = tensor_x)
        tensor_x = tensor_x.view(len(datasets_dirichlet_train_x[i]), 1, 28, 28)
        tensor_y = torch.Tensor(datasets_dirichlet_train_y[i])
        tensor_y = tensor_y.long()

        dataset_train = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
        dataloader_train[i] = torch.utils.data.DataLoader(dataset_train, batch_size = 64) # create your dataloader
    
    device = "cuda"
    ## You can play with these parameters & observe how they effect the training process.
    learning_rate = 0.005
    epochs = 100
    base_net = None
    base_net = Net()
    
    ## instantiate your model

    ## create two dataloaders (train & test) to load data into batches   
    #train_datatset, test_dataset = prepare_datasets()
    ## instantiate an optimizer for you model, as well as a criterion/loss function
    print("Get dataloader")
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size = 128)
    print("Finish getting dataloader")
   

    # Evaluating the model before any training steps is a good practice
    #accuracy, loss = test(net_dict[, criterion, testloader)
    #print('Before training :',f'Accuracy: {accuracy}', f'Loss: {loss}' )
    
    time0 = time() 
    directory = "/home/skerres/Studium/Cloud_Computing/FederatedLearning/weights_mnist"
    date_time = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    save_directory = os.path.join(directory, date_time)
    os.mkdir(save_directory)
    for epoch in range(epochs):
        active_clients = np.random.choice(np.arange(10), 10, replace=False) 
        net_dict = {}
        for i in active_clients:
            net_dict[i] = copy.deepcopy(base_net)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(net_dict[i].parameters(), lr=learning_rate)
            trainloader_client = dataloader_train[i]
            train(epoch, net_dict[i], optimizer, criterion, trainloader_client)
            accuracy, loss = test(net_dict[i], criterion, dataloader_test)
            print(f'Epoch :{epoch}', f'Client: {i}',f'Accuracy: {accuracy}', f'Loss: {loss}' )
            if epoch % 10 == 0 and epoch > 0:
                filename = "client_" + str(i) + "_epoch_" + str(epoch) + ".pt"
                filepath = os.path.join(save_directory, filename)
                torch.save(net_dict[i].state_dict(), filepath)
    
        base_net = aggregrate_models(net_dict, base_net, active_clients)
        accuracy, loss = test(base_net, criterion, dataloader_test)

        print(f'Epoch :{epoch}',f'Accuracy: {accuracy}', f'Loss: {loss}' )
    print(f'Training Time (minutes) :{(time()-time0) / 60}')

def aggregrate_models(net_dict, base_net, active_clients): 
    update_state = {}
    first = True
    for k in active_clients:
        # print(update_state["conv1.weight"][0])
        for key in net_dict[k].state_dict().keys():
            # print("Updating weights for key: " + str(key) + " and client: " + str(k))
            if first:
                update_state[key] = net_dict[k].state_dict()[key]
            else:
                update_state[key] += net_dict[k].state_dict()[key]
        first = False
    for key in update_state:
         update_state[key] = update_state[key] / len(active_clients)
    base_net.load_state_dict(copy.deepcopy(update_state))
    return base_net

if __name__ == "__main__":
    main() 

