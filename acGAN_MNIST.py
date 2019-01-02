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
##  Impelementation of cGAN on MNIST dataset
##  I referred to following examples to develop this sample :
##  https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/generative_adversarial_network/main.py
##  https://github.com/TuXiaokang/ACGAN.PyTorch/blob/master/main.py
######################################################################################################################################
######################################################################################################################################

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Choose GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device " , device)

# Change here learning parameters
batch_size = 50
latent_size = 64    
hidden_size = 256
image_size = 784
lr = 0.0002
nb_classes = 10
num_epochs = 50 # Results become interesting from epoch ~20

resul_dir = 'ResulacGan'
if not os.path.exists(resul_dir):
    os.makedirs(resul_dir)


dataTransform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                     std=(0.5, 0.5, 0.5))])
# MNIST Train data set
train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist_data', 
                                                          download=True, 
                                                          train=True,
                                                          transform=dataTransform), 
                                           batch_size=batch_size, 
                                           shuffle=True)


#Discriminator model
class Discriminator(nn.Module):
    def __init__(self , input_size, hidden_size, output_size):
         super(Discriminator, self).__init__()
         self.linear1 = nn.Linear(input_size , hidden_size)
         self.linear2 = nn.Linear(hidden_size , hidden_size)
         self.linear3 = nn.Linear(hidden_size, output_size)
         self.linearAUx = nn.Linear(hidden_size , nb_classes)
     #image and label
    def forward(self, x ):
        x = F.leaky_relu(self.linear1(x), 0.2) 
        x = F.leaky_relu(self.linear2(x), 0.2)
        output = torch.sigmoid((self.linear3(x)))
        aux = self.linearAUx(x)
        return output, aux

# Generator Model
class Generator(nn.Module):

    def __init__(self , input_size, hidden_size, output_size):
         super(Generator, self).__init__()
         self.linear1 = nn.Linear(input_size, hidden_size)
         self.linear2 = nn.Linear(hidden_size , hidden_size)
         self.linear3 = nn.Linear(hidden_size, output_size)
         self.label_embedding = nn.Embedding(nb_classes, latent_size)
    # x random  y labels
    def forward(self, x, y):
        x  = torch.mul(self.label_embedding(y), x)
        #x = torch.cat((self.label_embedding(y), x), -1)
        x = F.relu(self.linear1(x)) 
        x = F.relu(self.linear2(x))
        x= self.linear3(x)
        return torch.tanh(x) #Tied Sigmoid instead : did not work


#initialize discriminator and generator
D = Discriminator(image_size, hidden_size ,1).to(device) ;
G = Generator(latent_size , hidden_size,image_size).to(device)

#Adam optimization 
optimizerD = torch.optim.Adam(D.parameters(), lr)
optimizerG = torch.optim.Adam(G.parameters(), lr)
criterion_output = nn.BCELoss()
criterion_aux = nn.CrossEntropyLoss() 



total_step = len(train_loader)

for epoch in range(num_epochs):
    for batch_idx, (x, target) in enumerate(train_loader):

        images = x.reshape(batch_size, -1).to(device)
        realLabel = torch.ones(batch_size, 1).to(device)
        fakeLabel = torch.zeros(batch_size, 1).to(device)
       
        target = torch.LongTensor(target).to(device)
        
        # TRAIN D
        # On true data
        predictR, predictRLabel = D(images) #image from the real dataset
        loss_real_output = criterion_output(predictR, realLabel)  # compare vs label =1 (D is supposed to "understand" that the image is real)
        loss_real_aux = criterion_aux(predictRLabel , target)
        real_score = predictR

        # On fake data
        latent_value = torch.randn((batch_size, latent_size)).to(device)
        gen_labels = torch.LongTensor(np.random.randint(0, nb_classes, batch_size)).to(device)
       
        fake_images = G(latent_value , gen_labels) #generate a fake image
        predictF, predictLabelF = D(fake_images)
        loss_fake_output = criterion_output(predictF , fakeLabel) # compare vs label =0 (D is supposed to "understand" that the image generated by G is fake)
        loss_fake_aux = criterion_aux(predictLabelF, gen_labels)
        fake_score = predictF

        lossD = loss_real_output + loss_real_aux + loss_fake_output +loss_fake_aux

        optimizerD.zero_grad() 
        optimizerG.zero_grad() 
        lossD.backward()
        optimizerD.step() 
        
        # TRAIN G
        latent_value = torch.randn((batch_size, latent_size)).to(device)
        gen_labels = torch.LongTensor(np.random.randint(0, nb_classes, batch_size)).to(device)
        fake_images= G(latent_value , gen_labels) #Generate a fake image
        predictG, predictLabelG = D(fake_images)
        lossG_output = criterion_output(predictG , realLabel) # Compare vs label = 1 (We want to optimize G to fool D, predictG must tend to 1)
        lossG_aux = criterion_aux( predictLabelG, gen_labels)
        lossG = lossG_output+ lossG_aux
        optimizerD.zero_grad() 
        optimizerG.zero_grad() 
        lossG.backward()
        optimizerG.step() 

        if (batch_idx+1) % 200 == 0:
            print("Epoch: "+str(epoch)+"/"+str(num_epochs)+ "  -- Batch:"+ str(batch_idx+1)+"/"+str(total_step))
            print("     GenLoss "+str(round(lossG.item(), 3))+ "  --  DiscLoss "+str(round(lossD.item(), 3)))
            print("     D(x): "+str(round(real_score.mean().item(), 3))+ "  -- D(G(z)):"+str(round(fake_score.mean().item(), 3)))

    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(resul_dir, 'real_images.png'))
    
    # Save sampled images
    with torch.no_grad() :
        latent_value = torch.randn((60, latent_size)).to(device)
        gen_labels = np.zeros(0)
        for i in range(nb_classes):
            gen_labels = np.append(gen_labels, np.ones(6)*i )
        gen_labels = torch.from_numpy(gen_labels)
        gen_labels = gen_labels.type(torch.LongTensor).to(device)
        fake_images = G(latent_value , gen_labels) #generate a fake image
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images), os.path.join(resul_dir, 'fake_images-{}.png'.format(epoch+1)))
    

# generate samples for all labels
nbImageToGenerate = 8*8
for i in range(10):
    latent_value = torch.randn((nbImageToGenerate, latent_size)).to(device)
    gen_labels = torch.LongTensor(np.full(nbImageToGenerate , i )).to(device)
    fake_images = G(latent_value , gen_labels) #Generate a fake image
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(resul_dir, 'GeneratedSample-{}.png'.format(i)))