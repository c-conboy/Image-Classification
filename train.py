
#import math
import torchvision.transforms as transforms 
from torchvision.utils import save_image
import net as net
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import os
import argparse

#Argument parsing:
parser = argparse.ArgumentParser()

# training options
parser.add_argument('-s', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('-l', type=str, default='./frontendsaved100.pth')
parser.add_argument('-lr', type=float, default=1e-5)
parser.add_argument('-e', type=int, default=50)
parser.add_argument('-b', type=int, default=1000)

parser.add_argument('-cuda', type=str, default='Y')

parser.add_argument('-p', type=str, default='loss100.png')
args = parser.parse_args()


epochs = args.e
batchs = args.b
if(args.cuda=='Y'):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


train_transform = transforms.Compose([transforms.ToTensor()])
train_set = CIFAR100('./data/', train=True, download=True, transform=train_transform)
train_dataloader = DataLoader(train_set, batch_size=int(batchs), shuffle=True)

encoder = net.encoder_decoder.encoder
encoder.load_state_dict(torch.load("./encoder.pth", map_location=device))
#frontend = net.encoder_decoder.frontend
#frontend.load_state_dict(torch.load("./frontend1.pth", map_location=device))
#model = net.CJNet(encoder, frontend)

model = net.CJNet(encoder)
model.train()
model.to(device)

optimizer = torch.optim.Adam(model.frontend.parameters(), lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss()

#Tensor to hold loss at each epoch
losses_data = torch.zeros(epochs)
losses_data = losses_data.to(torch.device(device))
for i in tqdm(range(epochs)):
    print("epoch:" + str(i))
    k = -1
    for batch in train_dataloader:
        #start recording loss
        k+=1
        print('i, batchNumber', i, k)
        batch_loss = 0
        for j in range(batchs):
            image = batch[0][j]
            image = image.to(device)
            #call forward function
            inference = model(image)
            target = batch[1][j]
            target = target.to(device)
            
            #Calculate loss
            cross_entropy_loss = loss_fn(inference, target)
            #Add to batch loss
            batch_loss += cross_entropy_loss
        #update model parameters based on loss
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        #update epoch loss
        batch_loss = batch_loss/batchs
        print(batch_loss)
    if(i%10 == 0 or i == (epochs-1)):
        torch.save(model.frontend.state_dict(), './frontend100_iter_{:d}.pth'.format(i + 1))
    losses_data[i] += batch_loss


torch.save(model.frontend.state_dict(), args.l)
plt.plot(losses_data.cpu().detach().numpy())

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("./loss100.png")