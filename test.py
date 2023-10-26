import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms 
from torchvision.utils import save_image
import net as net
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


train_transform = transforms.Compose([transforms.ToTensor()])
train_set = CIFAR10('./data/', train=False, download=True, transform=train_transform)
train_dataloader = DataLoader(train_set, batch_size=int(4), shuffle=True)

encoder = net.encoder_decoder.encoder
encoder.load_state_dict(torch.load("./encoder.pth", map_location='cpu'))
model = net.CJNet(encoder)
model.eval()
out_tensor = model(train_set[0][0])
print(out_tensor)