import torch

def count_parameters(model):
	return(sum(p.numel() for p in model.parameters() if p.requires_grad))

import torchvision.models as models

net = models.alexnet(pretrained=True)
net.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
print(count_parameters(net))