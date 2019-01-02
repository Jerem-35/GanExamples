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
##  https://github.com/TuXiaokang/ACGAN.PyTorch/blob/master/main.py
## 
######################################################################################################################################
######################################################################################################################################


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class Generator(nn.Module):

    def __init__(self, latent_size , nb_filter, nb_classes):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(nb_classes, latent_size)
        self.conv1 = nn.ConvTranspose2d(latent_size, nb_filter * 8, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(nb_filter * 8)
        self.conv2 = nn.ConvTranspose2d(nb_filter * 8, nb_filter * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(nb_filter * 4)
        self.conv3 = nn.ConvTranspose2d(nb_filter * 4, nb_filter * 2, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(nb_filter * 2)
        self.conv4 = nn.ConvTranspose2d(nb_filter * 2, nb_filter * 1, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(nb_filter * 1)
        self.conv5 = nn.ConvTranspose2d(nb_filter * 1, 3, 4, 2, 1)
        self.__initialize_weights()

    def forward(self, input, cl):
        x = torch.mul(self.label_embedding(cl), input)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        return torch.tanh(x)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class Discriminator(nn.Module):

    def __init__(self, nb_filter, num_classes=10):
        super(Discriminator, self).__init__()
        self.nb_filter = nb_filter
        self.conv1 = nn.Conv2d(3, nb_filter, 4, 2, 1)
        self.conv2 = nn.Conv2d(nb_filter, nb_filter * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(nb_filter * 2)
        self.conv3 = nn.Conv2d(nb_filter * 2, nb_filter * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(nb_filter * 4)
        self.conv4 = nn.Conv2d(nb_filter * 4, nb_filter * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(nb_filter * 8)
        self.conv5 = nn.Conv2d(nb_filter * 8, nb_filter * 1, 4, 1, 0)
        self.gan_linear = nn.Linear(nb_filter * 1, 1)
        self.aux_linear = nn.Linear(nb_filter * 1, num_classes)
        self.__initialize_weights()
    
    def forward(self, input):
        x = self.conv1(input)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv5(x)
        x = x.view(-1, self.nb_filter * 1)
        c = self.aux_linear(x)
        s = self.gan_linear(x)
        s = torch.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device " , device)
batch_size = 100
latent_size = 100
lr = 0.0002
num_epochs = 70
nb_classes = 10 

resul_dir = 'ResulACGanCifar'
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



D = Discriminator(64, nb_classes).to(device)
G = Generator(latent_size, 64 ,  nb_classes).to(device)


#Adam optimization 
optimizerD = torch.optim.Adam(D.parameters(), lr, betas = (0.5, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr, betas = (0.5, 0.999))

criterion_adv = nn.BCELoss()
criterion_aux = nn.CrossEntropyLoss() 

total_step = len(train_loader)
for epoch in range(num_epochs):
    for batch_idx, (x, target) in enumerate(train_loader):
        images = x.to(device)
        
        current_batchSize = images.size()[0] 
        
        realLabel = torch.ones(current_batchSize).to(device)
        fakeLabel = torch.zeros(current_batchSize).to(device)

        target = torch.LongTensor(target).to(device)
        
        # TRAIN D
        # On true data
        predictR, predictRLabel = D(images) #image from the real dataset
        loss_real_adv = criterion_adv(predictR, realLabel)  # compare vs label =1 (D is supposed to "understand" that the image is real)
        loss_real_aux = criterion_aux(predictRLabel , target)
        real_score = predictR
        
        # On fake data
        latent_value = torch.randn(current_batchSize, latent_size).to(device)
        gen_labels = torch.LongTensor(np.random.randint(0, nb_classes, current_batchSize)).to(device)
        fake_images = G(latent_value , gen_labels) #generate a fake image
        predictF, predictFLabel = D(fake_images)
        loss_fake_adv = criterion_adv(predictF ,  fakeLabel) # compare vs label =0 (D is supposed to "understand" that the image generated by G is fake)
        loss_fake_aux = criterion_aux(predictFLabel, gen_labels)
        fake_score = predictF
        
        lossD = loss_real_adv + loss_real_aux  +loss_fake_adv + loss_fake_aux
        
        optimizerD.zero_grad() 
        optimizerG.zero_grad() 
        lossD.backward()
        optimizerD.step() 
        
        # TRAIN G
        latent_value = torch.randn(current_batchSize, latent_size).to(device)
        gen_labels = torch.LongTensor(np.random.randint(0, nb_classes, current_batchSize)).to(device)
        fake_images= G(latent_value, gen_labels) #Generate a fake image
        predictG, predictLabel = D(fake_images)
        lossG_adv = criterion_adv(predictG, realLabel) # Compare vs label = 1 (We want to optimize G to fool D, predictG must tend to 1)
        lossG_aux = criterion_aux(predictLabel, gen_labels)
        lossG = lossG_adv + lossG_aux
        optimizerD.zero_grad() 
        optimizerG.zero_grad() 
        lossG.backward()
        optimizerG.step() 

        if (batch_idx+1) % 100 == 0:
            print("Epoch: "+str(epoch)+"/"+str(num_epochs)+ "  -- Batch:"+ str(batch_idx+1)+"/"+str(total_step))
            print("     GenLoss "+str(round(lossG.item(), 3))+ "  --  DiscLoss "+str(round(lossD.item(), 3)))
            print("     D(x): "+str(round(real_score.mean().item(), 3))+ "  -- D(G(z)):"+str(round(fake_score.mean().item(), 3)))

    with torch.no_grad():
        fake_images = fake_images.reshape(fake_images.size(0), 3, 64, 64)
        save_image(denorm(fake_images), os.path.join(resul_dir, 'fake_images-{}.png'.format(epoch+1)))
    if (epoch+1) == 1:
        save_image(images, os.path.join(resul_dir, 'real_images.png'),  normalize = True)


nbImageToGenerate = 8*8
for i in range(10):
    latent_value = torch.randn((nbImageToGenerate, latent_size)).to(device)
    gen_labels = torch.LongTensor(np.full(nbImageToGenerate , i )).to(device)
    fake_images = G(latent_value , gen_labels) #Generate a fake image
    fake_images = fake_images.reshape(fake_images.size(0), 3, 64, 64)
    save_image(denorm(fake_images), os.path.join(resul_dir, 'GeneratedSample-{}.png'.format(i)))


#torch.save(G.state_dict(),"trainining_acgancifar.pt")

