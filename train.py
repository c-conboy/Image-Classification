
import torchvision.transforms as transforms 
from torchvision.utils import save_image
import net as net
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch

if(1):
    device = torch.device('cpu')
else:
    device = torch.device('cpu')




train_transform = transforms.Compose([transforms.ToTensor()])
train_set = CIFAR10('./data/', train=True, download=True, transform=train_transform)
train_dataloader = DataLoader(train_set, batch_size=int(8), shuffle=True)


encoder = net.encoder_decoder.encoder
encoder.load_state_dict(torch.load("./encoder.pth", map_location='cpu'))
model = net.CJNet(encoder)
model.train()
model.to(device)




optimizer = torch.optim.Adam(model.frontend.parameters(), lr=1e-4)

for i in tqdm(range(10)):
    #Tensor to hold loss at each epoch
    for batch in train_dataloader:
        #start recording loss
        for img in batch:
            #call forward function
            #Calculate error
            #Add to batch loss
    #update model parameters based on loss


