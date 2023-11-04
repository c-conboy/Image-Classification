
#import math
import torchvision.transforms as transforms 
from torchvision.utils import save_image
import net as net
from torchvision.datasets import CIFAR10
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
parser.add_argument('-l', type=str, default='frontendsaved.pth')
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-e', type=int, default=20)
parser.add_argument('-b', type=int, default=100)

parser.add_argument('-cuda', type=str, default='N')

parser.add_argument('-p', type=str, default='loss.png')
args = parser.parse_args()



epochs = 20
batchs = 100
if(0):
    device = torch.device('cuda')
else:
    device = torch.device('cuda')


train_transform = transforms.Compose([transforms.ToTensor()])
train_set = CIFAR10('./data/', train=True, download=True, transform=train_transform)
train_dataloader = DataLoader(train_set, batch_size=int(batchs), shuffle=True)

encoder = net.encoder_decoder.encoder
encoder.load_state_dict(torch.load("./encoder.pth", map_location=device))
#frontend = net.encoder_decoder.frontend
#frontend.load_state_dict(torch.load("./frontend1.pth", map_location=device))
#model = net.CJNet(encoder, frontend)

model = net.CJNet(encoder)
model.train()
model.to(device)

optimizer = torch.optim.Adam(model.frontend.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

#Tensor to hold loss at each epoch
losses_data = torch.zeros(epochs)
losses_data = losses_data.to(torch.device('cuda'))
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
        state_dict = model.frontend.state_dict()
        torch.save(state_dict, 
                   './frontend_iter_{:d}.pth.tar'.format(i + 1))
    losses_data[i] += batch_loss

state_dict = model.frontend.state_dict()
decoder_state_dict_file = os.path.join(os.getcwd(), "/frontend.pth")
torch.save(state_dict, decoder_state_dict_file)

plt.plot(losses_data.cpu().detach().numpy())

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("./loss.png")