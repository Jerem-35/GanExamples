import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

######################################################################################################################################
######################################################################################################################################
##
##  Impelementation of GAN on CIFAR dataset
##  I referred to following example to develop this sample :
## https://becominghuman.ai/understanding-and-building-generative-adversarial-networks-gans-8de7c1dc0e25
## 
######################################################################################################################################
######################################################################################################################################

#Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
         super(Discriminator, self).__init__()
         self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
         self.batch1 = nn.BatchNorm2d(64)
         self.conv2 = nn.Conv2d(64, 128, 4, 2 , 1)
         self.batch2 = nn.BatchNorm2d(128)
         self.conv3= nn.Conv2d(128 ,256 , 4, 2 , 1)
         self.batch3 = nn.BatchNorm2d(256)
         self.conv4 = nn.Conv2d(256, 512, 4, 2 , 1)
         self.batch4 = nn.BatchNorm2d(512)
         self.conv5 = nn.Conv2d(512 , 1, 4) 

    def forward(self, x):
        x =  self.batch1(self.conv1(x))
        x = F.leaky_relu(x, 0.2)
        x = self.batch2(self.conv2(x))
        x = F.leaky_relu(x, 0.2)
        x = self.batch3(self.conv3(x))
        x = F.leaky_relu(x, 0.2)
        x = self.batch4(self.conv4(x))
        x = F.leaky_relu(x, 0.2)
        return torch.sigmoid(self.conv5(x)).view(-1)

class Generator(nn.Module):
    def __init__(self, latent_size):
         super(Generator, self).__init__()
         self.deconv1 = nn.ConvTranspose2d(latent_size, 512, 4, 1, 0)
         self.batch1 = nn.BatchNorm2d(512)
         self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2 , 1)
         self.batch2 = nn.BatchNorm2d(256)
         self.deconv3= nn.ConvTranspose2d(256 ,128 , 4, 2 , 1)
         self.batch3 = nn.BatchNorm2d(128)
         self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2 , 1)
         self.batch4 = nn.BatchNorm2d(64)
         self.deconv5 = nn.ConvTranspose2d(64 , 3, 4, 2 , 1) 

    def forward(self, x):
        x =  self.batch1(self.deconv1(x))
        x = F.relu(x)
        x = self.batch2(self.deconv2(x))
        x = F.relu(x)
        x = self.batch3(self.deconv3(x))
        x = F.relu(x)
        x = self.batch4(self.deconv4(x))
        x = F.relu(x)
        return torch.tanh(self.deconv5(x))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device " , device)

batch_size = 64
latent_size = 100
lr = 0.0002
num_epochs = 100

resul_dir = 'ResulGanCifar'
if not os.path.exists(resul_dir):
    os.makedirs(resul_dir)

dataTransform=transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('cifar_data', 
                                                          download=True, 
                                                          train=True,
                                                          transform=dataTransform), 
                                           batch_size=batch_size, 
                                           shuffle=True)
# about weight initialization https://discuss.pytorch.org/t/weight-initilzation/157/14 
def WeightInitialization(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
       nn.init.xavier_uniform(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

D = Discriminator().to(device)
WeightInitialization(D)
G = Generator(latent_size).to(device)
WeightInitialization(G)

#Adam optimization 
optimizerD = torch.optim.Adam(D.parameters(), lr, betas = (0.5, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr, betas = (0.5, 0.999))

# Binary cross entropy loss
criterion = nn.BCELoss()


total_step = len(train_loader)
for epoch in range(num_epochs):
    for batch_idx, (x, target) in enumerate(train_loader):
        images = x.to(device)


        realLabel = torch.ones(images.size()[0]).to(device)
        fakeLabel = torch.zeros(images.size()[0]).to(device)
       
        
        # TRAIN D
        # On true data
        predictR = D(images) #image from the real dataset
        loss_real = criterion(predictR, realLabel)  # compare vs label =1 (D is supposed to "understand" that the image is real)
        real_score = predictR

        # On fake data
        latent_value = torch.randn((images.size()[0], latent_size, 1 ,1)).to(device)
        fake_images = G(latent_value) #generate a fake image
        predictF = D(fake_images)
        loss_fake = criterion(predictF ,  fakeLabel) # compare vs label =0 (D is supposed to "understand" that the image generated by G is fake)
        fake_score = predictF

        lossD = loss_real + loss_fake 

        optimizerD.zero_grad() 
        optimizerG.zero_grad() 
        lossD.backward()
        optimizerD.step() 
        
        # TRAIN G
        latent_value = torch.randn((images.size()[0], latent_size, 1 ,1)).to(device)
        fake_images= G(latent_value) #Generate a fake image
        predictG = D(fake_images)
        lossG = criterion(predictG, realLabel) # Compare vs label = 1 (We want to optimize G to fool D, predictG must tend to 1)
        optimizerD.zero_grad() 
        optimizerG.zero_grad() 
        lossG.backward()
        optimizerG.step() 

        if (batch_idx+1) % 100 == 0:
            print("Epoch: "+str(epoch)+"/"+str(num_epochs)+ "  -- Batch:"+ str(batch_idx+1)+"/"+str(total_step))
            print("     GenLoss "+str(round(lossG.item(), 3))+ "  --  DiscLoss "+str(round(lossD.item(), 3)))
            print("     D(x): "+str(round(real_score.mean().item(), 3))+ "  -- D(G(z)):"+str(round(fake_score.mean().item(), 3)))

        if batch_idx == 700:
             # Save sampled images
            save_image(fake_images, os.path.join(resul_dir, 'fake_images-{}.png'.format(epoch+1)) , normalize = True)
            # Save real images
            if (epoch+1) == 1:
                save_image(images, os.path.join(resul_dir, 'real_images.png'),  normalize = True)
    
