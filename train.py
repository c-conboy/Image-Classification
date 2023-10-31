
import math
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
import os

epochs = 2
batchs = 100
if(1):
    device = torch.device('cpu')
else:
    device = torch.device('cpu')

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = CIFAR10('./data/', train=True, download=True, transform=train_transform)
train_dataloader = DataLoader(train_set, batch_size=int(batchs), shuffle=True)

encoder = net.encoder_decoder.encoder
encoder.load_state_dict(torch.load("./encoder.pth", map_location='cpu'))
model = net.CJNet(encoder)
model.train()
model.to(device)

optimizer = torch.optim.Adam(model.frontend.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

#Tensor to hold loss at each epoch
losses_data = torch.zeros(epochs)
for i in tqdm(range(epochs)):
    print("epoch:" + str(i))
    for batch in train_dataloader:
        #start recording loss
        batch_loss = 0
        for i in range(batchs):
            image = batch[0][i]
            image = image.to(device)
            #call forward function
            inference = model(image)
            #Calculate loss
            cross_entropy_loss = loss_fn(inference, batch[1][i])
            #Add to batch loss
            batch_loss += cross_entropy_loss
        #update model parameters based on loss
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        #update epoch loss
        batch_loss = batch_loss/batchs
        print(batch_loss)
    losses_data[i] += batch_loss



state_dict = model.frontend.state_dict()
decoder_state_dict_file = os.path.join(os.getcwd(), "/frontend.pth")


plt.plot(losses_data.detach().numpy())

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("./loss.png")