#%%
import os
import cv2
import torch
import gc
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from myGAN_nogpt import *
from mylosses import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import random_split
import itertools

# data transform
transform_random = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(15),
])

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5],[0.5]),
])

transform_ = transforms.Compose([
    transforms.ToTensor(),
])

# constants
dataset =  '../datasets/신의탑_DB/' #'../datasets/my_train/'
val_data = '../datasets/my_validation/' #'../datasets/webtoon/webtoon_images/'
store = './epoch_fixed_gen/'
test_data = '../datasets/my_test/'



# function to load image
def load_image(image_file):
    img = Image.open(image_file).convert('RGB').resize((x_shape, y_shape), Image.BICUBIC)
    #img = transform_random(img)
    img_gray = img.convert('L').resize((x_shape, y_shape), Image.BICUBIC)
    
    return transform_(img), transform(img_gray)


# constants
epochs = 700
x_shape = 512
y_shape = 512
fixed_seed_num = 123
torch.manual_seed(fixed_seed_num)

# initialize the cGAN model with (generator, discriminator)
netG = Generator().to('cuda')
netD = Discriminator().to('cuda')
cGAN = CGAN(netG, netD).to('cuda')

# compile with custom loss functions
# &&& We'll figure it out &&&
optim_G = torch.optim.Adam(netG.parameters(), lr=1E-4, betas=(0.9, 0.999), eps=1e-08)
optim_D = torch.optim.Adam(netD.parameters(), lr=1E-4, betas=(0.9, 0.999), eps=1e-08)


# &&& We'll figure it out &&&
criterion = nn.BCELoss()
L1loss = nn.L1Loss()

# weights_gen = torch.load('weights_gen.pt')
# weights_disc = torch.load('weights_disc.pt')
# netG.load_state_dict(weights_gen)
# netD.load_state_dict(weights_disc)

# Load the first 100 files
load_data_size = 600
file_list = os.listdir(dataset)
file_list = list(itertools.islice(file_list, load_data_size))

# Load the datasets once before the training loop
dataset_ = [(load_image(os.path.join(dataset, file))) for file in file_list]

total_size = len(dataset_)
train_size = int(0.8 * total_size) # 80% for training
valid_size = int(0.1 * total_size) # 10% for validation
test_size = total_size - train_size - valid_size

train_dataset, val_dataset_, test_dataset_ = random_split(dataset_, [train_size, valid_size, test_size])


# Data loaders
train_dataloader = DataLoader(dataset_, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset_, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset_, batch_size=1, shuffle=True)

patience = 500  # Number of epochs with no improvement after which training will be stopped.
num_epochs_no_improvement = 0
best_val_loss = np.inf
total_val_loss = np.inf

G_losses = np.array([])
D_losses = np.array([])
cur_G_losses = np.array([])
cur_D_losses = np.array([])
training_reconstruction_losses = np.array([])
validation_reconstruction_losses = np.array([])
cur_epochs = []

netD.apply(weights_init)
netG.apply(weights_init)

# train the cGAN model with specified number of epochs
for e in range(epochs):
    cur_G_losses = np.array([])
    cur_D_losses = np.array([])
    cur_training_rec_loss = np.array([])
    cur_epochs.append(e)
    for i, (real_img, gray_img) in enumerate(train_dataloader):
        # print(f'Epoch {e}, disc_Batch {i}')
        weight_d: int = 5
        weight_g: int = 1
        # Transfer data to GPU
        real_img, gray_img = real_img.to('cuda'), gray_img.to('cuda')
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        
        ## training mode with G, D
        netD.train() # D Neural Network
        netG.train() # G Neural Network
        ## Train Discriminator
        # Generate fake image batch with G
        fake_img = netG(gray_img)
        
        # disc의 파라미터
        input = torch.cat([gray_img, gray_img], dim=0)
        output = torch.cat([real_img, fake_img.detach()], dim=0)
        
        # labels are used for Discriminator as y_true
        labels = torch.cat((0.7 + torch.rand(1, 1) * 0.3, torch.rand(1, 1) * 0.3))
        # 정답 label 값 0.7~ 1 사이 , fake label : 0 ~ 0.3
        # discriminator한테 혼동주기 위해 사용
        labels = labels.to('cuda')

        disc_out = netD(input, output)
        errD = criterion(disc_out, labels)
        
        cur_D_losses = np.append(cur_D_losses, weight_d * errD.item())
        
        optim_D.zero_grad()
        errD.backward()
        # D_G_z2 = output.mean().item()
        optim_D.step()
        
        print(disc_out)
        
        

        #g_out = netG(gray_img) -> 주석처리함, 위에 생성한 fake_img 활용
        d_out = netD(gray_img, fake_img) # g_out -> fake_img
        d_loss = criterion(d_out, torch.ones(1, 1).to('cuda'))
        g_loss = L1loss(fake_img, real_img) # g_out -> fake_img
        # g_loss = criterion(g_out, real_img)
        
        errG =  (weight_d * d_loss + weight_g * g_loss)
        
        cur_G_losses = np.append(cur_G_losses, errG.item())
        cur_training_rec_loss = np.append(cur_training_rec_loss, weight_g * g_loss.item())
        
        # print("OMGOMASDFOMASDOFMAOSMDFO", d_loss.item(), " ", g_loss.item(), " ",errG.item())
        
        optim_G.zero_grad()
        errG.backward()
        optim_G.step()
        
    D_losses = np.append(D_losses, cur_D_losses.mean())
    G_losses = np.append(G_losses, cur_G_losses.mean())
    training_reconstruction_losses = np.append(training_reconstruction_losses, cur_training_rec_loss.mean())
        
    save_image(real_img, f'{store}epoch{e}_real_.jpg')
    save_image(fake_img, f'{store}epoch{e}_train_.jpg')
    # Validate
    netG.eval()
    with torch.no_grad():
        total_val_loss = 0
        for i, (real_img, gray_img) in enumerate(val_dataloader):
            real_img, gray_img = real_img.to('cuda'), gray_img.to('cuda')
            fake_img_val = netG(gray_img)
            val_loss = L1loss(fake_img_val, real_img).item()
            # val_loss = criterion(fake_img_val, real_img).item()
            total_val_loss += val_loss
        avg_val_loss = total_val_loss / len(val_dataloader)
        print("average test loss : ", avg_val_loss)
        validation_reconstruction_losses = np.append(validation_reconstruction_losses, avg_val_loss)
        save_image(fake_img_val, f'{store}epoch{e}_val_.jpg')
    print(f'Average val loss: {avg_val_loss}')
    
    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        num_epochs_no_improvement = 0
        # torch.save(gen.state_dict(), 'best_weights_gen.pt')
        # torch.save(disc.state_dict(), 'best_weights_disc.pt')
    else:
        num_epochs_no_improvement += 1
        if num_epochs_no_improvement == patience:
            print("Early stopping")
            # torch.save(gen.state_dict(), 'weights_gen.pt')
            # torch.save(disc.state_dict(), 'weights_disc.pt')
            break
    # After the training loop
    netG.eval()
    with torch.no_grad():
        total_test_loss = 0
        for i, (real_img, gray_img) in enumerate(test_dataloader):
            real_img, gray_img = real_img.to('cuda'), gray_img.to('cuda')
            gen_image_test = netG(gray_img)
            test_loss = L1loss(gen_image_test, real_img).item()
            # test_loss = criterion(gen_image_test, real_img).item()
            total_test_loss += test_loss
        avg_test_loss = total_test_loss / len(test_dataloader)
        
        # entire_test_losses = np.append(entire_test_losses, avg_test_loss)
        save_image(gen_image_test, f'{store}epoch{e}_test_.jpg')
        save_image(gray_img, f'{store}epoch{e}_test_original_.jpg')
    print(f'Average test loss: {avg_test_loss}')
    plt.figure(figsize=(10, 6))
    plt.plot(cur_epochs, validation_reconstruction_losses, 'b', label='Validation Reconstruction error')
    plt.plot(cur_epochs, training_reconstruction_losses, 'r', label='Training Reconstruction error')
    plt.plot(cur_epochs, G_losses, 'violet', label='Generator loss')
    plt.plot(cur_epochs, D_losses, 'limegreen', label='Discriminator loss')
    plt.title('Validation and Test Errors')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    if e%10 == 0:
        torch.save(netG.state_dict(), 'weights_gen_initialize.pt')
        torch.save(netD.state_dict(), 'weights_disc_initialize.pt')
