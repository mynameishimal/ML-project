import pdb
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support

random.seed(0)
torch.manual_seed(0)
log_filename='network_wideresnet_squaredWeights'
writer  = SummaryWriter('covidDetectionRuns/'+log_filename)

data_path = 'dataset'
image_dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((600,700)),
        torchvision.transforms.ToTensor()
    ])
)
total_number = len(image_dataset)

trainset, testset = torch.utils.data.random_split(image_dataset, [int(.8*total_number), total_number -int(.8*total_number)])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4 ,
                                            shuffle = True, num_workers=0)

import torch.nn as nn
import torch.nn.functional as F


classes = image_dataset.class_to_idx
net = models.alexnet(pretrained=True)
net.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)

import torch.optim as optim
cost_function_weights = [1, 13993/2358]
criterion = nn.CrossEntropyLoss(torch.FloatTensor(cost_function_weights))
optimizer = optim.Adam(net.parameters(), weight_decay=0.5)
epoch_runs = 20

def add_performance(dictionary_performances, key, metric, size_limit=10):
    if key not in dictionary_performances:
        dictionary_performances[key] = []

    dictionary_performances[key].append(metric)
    if len(dictionary_performances[key])>size_limit: dictionary_performances[key].pop(0)
    return(dictionary_performances)

def has_best_performance(dictionary_performances, key, lowest, size_limit=10):
    if len(dictionary_performances[key])<size_limit: return False
    if mean(dictionary_performances['validation_loss']) < lowest:
        return(True)

validation_performance ={}

for epoch in range(epoch_runs):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        if (len(trainloader)*epoch + i) % 400 ==0:
            optimizer = optim.Adam(net.parameters(), weight_decay =.05)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # about to calculate gradients, set all to 0
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        writer.add_scalar('training loss', loss, len(trainloader)*epoch + i)
        loss.backward()
        optimizer.step() 

        y_true = labels  
        y_pred = torch.argmax(outputs, dim=1)
        trainingStats = precision_recall_fscore_support(y_true, y_pred, labels=(0,1), zero_division=0)

        
        for key, value in classes.items():
            writer.add_scalar(key + ' precision', trainingStats[0][value], len(trainloader)*epoch + i)
            writer.add_scalar(key + ' recall', trainingStats[1][value], len(trainloader)*epoch + i)
            writer.add_scalar(key + ' f score', trainingStats[2][value], len(trainloader)*epoch + i)

    	
     

        if i%10==0 or 1==1:
            validation_data = iter(testloader).next()
            v_inputs, v_labels = validation_data
            v_outputs = net(v_inputs)
            loss = criterion(v_outputs, v_labels)
            writer.add_scalar('validation loss', loss, len(trainloader)*epoch + i)

            v_true = v_labels
            v_pred = torch.argmax(v_outputs, dim=1)
            v_stats = precision_recall_fscore_support(v_true, v_pred, labels=(0,1 ), zero_division=0)
            validation_performance = add_performance(validation_performance, "validation_loss", loss)


            for key, value in classes.items():
                writer.add_scalar(key + 'validation precision', v_stats[0][value], len(trainloader)*epoch + i)
                validation_performance = add_performance(validation_performance, key +"_precision", v_stats[0][value])
                writer.add_scalar(key + 'validation recall', v_stats[1][value], len(trainloader)*epoch + i)
                validation_performance = add_performance(validation_performance, key + "_recall", v_stats[1][value])
                writer.add_scalar(key + 'validation f score', v_stats[2][value], len(trainloader)*epoch + i)
                validation_performance = add_performance(validation_performance, key +"_fscore", v_stats[2][value])

            

torch.save(net.state_dict(), 'models/same_as_tensorboard.model')
file_handlers = open('models/same_as_tensorboard_stats.txt', 'w')
file_handlers.write(' .... ')
file_handlers.close()




writer.close()
print('Finished Training')
f = open('finished_runs_log/'+log_filename+'.txt', 'a+')
f.write(log_filename+' has finished after '+str(len(trainloader)*epoch_runs)+' iterations')
f.close()

