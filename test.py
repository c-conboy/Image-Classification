import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms 
from torchvision.utils import save_image
import net as net
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


test_transform = transforms.Compose([transforms.ToTensor()])
test_set = CIFAR10('./data/', train=False, download=True, transform=test_transform)
test_dataloader = DataLoader(test_set, batch_size=int(4), shuffle=True)

frontend = net.encoder_decoder.frontend
frontend.load_state_dict(torch.load("./frontend.pth", map_location='cpu'))
model = net.CJNet(frontend)
model.eval()

top5err = 0
top1err = 0

for img in test_set:
    out_tensor = model(img[0])
    top5 = torch.topk(out_tensor, 5)
    for p in top5:
        if(out_tensor.tolist().index(p) == img[1]):
            top5err += 1
            continue

    top1 = torch.topk(out_tensor, 1)
    if(out_tensor.tolist().index(top1) == img[1]):
        top1err += 1
    
print(top5err/len(test_set))
print(top1err/len(test_set))