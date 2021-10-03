import torch
import pdb
def count_parameters(model):
	return(sum(p.numel() for p in model.parameters() if p.requires_grad))

import torchvision.models as models

net = models.alexnet(pretrained=True)
net.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
print("Alexnet: " +str(f'{count_parameters(net):,}'))
net = models.vgg16(pretrained=True)
net.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2, bias=True)
print("vgg16: "+str(f'{count_parameters(net):,}'))
net = models.resnet18(pretrained=True)
net.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
print("resnet18: "+str(f'{count_parameters(net):,}'))
net = models.wide_resnet50_2(pretrained=True)

net.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
print("wide_resnet50_2:" + str(f'{count_parameters(net):,}'))