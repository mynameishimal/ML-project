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
log_filename='resnet18_flip_modelBest'
writer  = SummaryWriter('covidDetectionWeightedBatchRuns/'+log_filename)
modelSave = SummaryWriter('models/'+log_filename)

data_path = 'dataset'
image_dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Resize((600,700)),
        torchvision.transforms.ToTensor()
    ])
)
total_number = len(image_dataset)

trainset, testset = torch.utils.data.random_split(image_dataset,
    [int(.8*total_number), total_number -int(.8*total_number)])

cost_function_weights = [1, (13993/2358)/2]
train_set_sample_weights = []
indices = []
for index in trainset.indices:
    pdb.set_trace()
    cur_image = trainset.dataset.imgs[index][1]
    indices.append(index)
    print(index)
    train_set_sample_weights.append(cost_function_weights[cur_image])
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights =
    train_set_sample_weights, num_samples=len(train_set_sample_weights))

test_set_sample_weights = []
for index in testset.indices:
    cur_image = testset.dataset.imgs[index][1]
    test_set_sample_weights.append(cost_function_weights[cur_image])
test_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights =
    test_set_sample_weights, num_samples=len(test_set_sample_weights))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=5,
    sampler=train_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size=5,
    sampler=test_sampler)

import torch.nn as nn
import torch.nn.functional as F


classes = image_dataset.class_to_idx

net = models.VGG16(pretrained=True)
pdb.set_trace()
net.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)

import torch.optim as optim
criterion = nn.CrossEntropyLoss(torch.FloatTensor(cost_function_weights))
optimizer = optim.Adam(net.parameters())
epoch_runs = 20

def add_performance(dictionary_performances, key, metric, size_limit=10):
    if key not in dictionary_performances:
        dictionary_performances[key] = []

    dictionary_performances[key].append(metric)
    if len(dictionary_performances[key])>size_limit: dictionary_performances[key].pop(0)
    return(dictionary_performances)

lowestAvg=1
lowest = lowestAvg

def has_best_performance(dictionary_performances, key, lowest, size_limit=10):
    if len(dictionary_performances[key])<size_limit: return((False, lowest))
    if mean(dictionary_performances[key]) < lowest:
        return(True,mean(dictionary_performances[key]))

validation_performance ={}

for epoch in range(epoch_runs):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):
        if (len(trainloader)*epoch + i) % 400 ==0:
            optimizer = optim.Adam(net.parameters())
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

        
     

        if i%10==0:
            validation_data = iter(testloader).next()
            v_inputs, v_labels = validation_data
            v_outputs = net(v_inputs)
            loss = criterion(v_outputs, v_labels)
            writer.add_scalar('validation loss', loss, len(trainloader)*epoch + i)

            v_true = v_labels
            v_pred = torch.argmax(v_outputs, dim=1)
            v_stats = precision_recall_fscore_support(v_true, v_pred, labels=(0,1 ), zero_division=0)
            validation_performance = add_performance(validation_performance, "validation_loss", loss.item())


            for key, value in classes.items():
                writer.add_scalar(key + 'validation precision', v_stats[0][value], len(trainloader)*epoch + i)
                validation_performance = add_performance(validation_performance, key +"_precision", v_stats[0][value])
                writer.add_scalar(key + 'validation recall', v_stats[1][value], len(trainloader)*epoch + i)
                validation_performance = add_performance(validation_performance, key + "_recall", v_stats[1][value])
                writer.add_scalar(key + 'validation f score', v_stats[2][value], len(trainloader)*epoch + i)
                validation_performance = add_performance(validation_performance, key +"_fscore", v_stats[2][value])

            is_best, lowest = has_best_performance(validation_performance, 'validation_loss', lowest)

            if is_best:
                torch.save(net.state_dict(), 'models/'+log_filename+'.model')
                file_handlers = open('models/'+log_filename+'.txt', 'w')
                file_handlers.write('validation loss:'+str(mean(validation_peformance['validation_loss'])))
                file_handlers.write('covid19 negative f1:'+str(mean(validaiton_peformance['covid19_negative_fscore'])))
                file_handlers.write('iterations: '+str(len(trainloader)*epoch +i))
                file_handlers.close()






writer.close()
print('Finished Training')
f = open('finished_runs_log/'+log_filename+'.txt', 'a+')
f.write(log_filename+' has finished after '+str(len(trainloader)*epoch_runs)+' iterations')
f.close()

