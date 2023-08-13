#%%
import itertools
import os

import torch

import matplotlib.pyplot as plt
import numpy as np

from typing import Callable

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from IPython.display import display
# from myGAN_heonwoo import *
from myGAN_nogpt import *

# constants
data_dir =  '../datasets/신의탑_DB/'
store = './epoch_test_variance/'

log_base_dir = "./logs"
training_title = "A"

save_every = 10

epochs = 100000 # 전체 epoch 횟수
batch_size = 1 # batch size
x_shape = 512 
y_shape = 512
fixed_seed_num = 1234

generator_lr = 1e-4 # learning rate of generator
discriminator_lr = 1e-4 # learning rate of discriminator

weight_d: int = 1
weight_g: int = 25

#weight initialization
#kaiming hee, xavier
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')

def load_image(image_path: str, x_shape: int, y_shape: int, transform: Callable):
    img_raw = Image.open(image_path)
    img_rgb = img_raw.convert('RGB').resize((x_shape, y_shape), Image.BICUBIC)
    img_gray = img_rgb.convert('L')
    return transform(img_rgb), transform(img_gray)

transform_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

transform = transforms.Compose([
    transforms.ToTensor(), # 0~1 scope
])

torch.manual_seed(fixed_seed_num)

generator = Generator().to('cuda')
discriminator = Discriminator().to('cuda')

generator_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=generator_lr,
)
discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=discriminator_lr
)
# discriminator_optimizer = torch.optim.SGD(
#     discriminator.parameters(),
#     lr=discriminator_lr,
#     momentum=0.9,
# )
# scheduler = torch.optim.lr_scheduler.CyclicLR(discriminator_optimizer, base_lr=1e-4, max_lr=0.01)

weights_gen = torch.load('weights_gen_addNoise.pt.pt')
weights_disc = torch.load('weights_disc_addNoise.pt.pt')
generator.load_state_dict(weights_gen)
discriminator.load_state_dict(weights_disc)

bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

load_data_size = 1000

file_list = os.listdir(data_dir)
file_list = list(itertools.islice(file_list, load_data_size))

dataset = [load_image(os.path.join(data_dir, file), x_shape, y_shape, transform) for file in file_list]

total_size = len(dataset)
train_size = int(0.8 * total_size)
valid_size = total_size - train_size

# train_paths = 
# val_paths = 
train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

log_dir = os.path.join(log_base_dir, training_title)
os.makedirs(log_dir, exist_ok=True)

summary_writer = SummaryWriter(log_dir)


patience = 5000  # Number of epochs with no improvement after which training will be stopped.
num_epochs_no_improvement = 0

best_val_loss = np.inf
total_val_loss = np.inf


training_reconstruction_losses = np.array([])
validation_reconstruction_losses = np.array([])
validation_adversarial_losses = np.array([])

adversarial_loss_means = np.array([])
reconstruction_loss_means = np.array([])
disc_loss_means = np.array([])
gen_loss_means = np.array([])

cur_epochs = []
# discriminator.apply(weights_init)
# generator.apply(weights_init)

# train the cGAN model with specified number of epochs
for e in range(epochs):
    print(f'Epoch {e}')
    cur_epochs.append(e)
    
    real_accuracies = np.array([])
    fake_accuracies = np.array([])
    
    disc_losses = np.array([])
    gen_losses = np.array([])
    adversarial_losses = np.array([])
    reconstruction_losses = np.array([])
    
    for i, (real_img, gray_img) in enumerate(train_dataloader):
        real_img, gray_img = real_img.to('cuda'), gray_img.to('cuda')
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        
        ## Train Discriminator
        # Generate fake image batch with G
        fake_img = generator(gray_img)
        
        # disc의 파라미터
        input = torch.cat([gray_img, gray_img], dim=0)
        output = torch.cat([real_img, fake_img.detach()], dim=0)
        
        # labels are used for Discriminator as y_true
        # 정답 label 값 0.7~ 1 사이 , fake label : 0 ~ 0.3
        # discriminator한테 혼동주기 위해 사용
        # labels = torch.cat((torch.ones(1, 1), torch.zeros(1, 1))).to('cuda')
        # labels = torch.cat((torch.ones(1, 1), torch.zeros(1, 1))).to('cuda')
        labels = torch.cat((0.7 + torch.rand(1, 1) * 0.3, torch.rand(1, 1) * 0.3)).to('cuda')
        
        disc_out = discriminator(input, output)
        disc_loss = bce_loss(disc_out, labels)
        
        real_accuracies = np.append(real_accuracies, (disc_out[:batch_size].detach().cpu() >= 0.5).float())
        fake_accuracies = np.append(fake_accuracies, (disc_out[batch_size:].detach().cpu() < 0.5).float())
        
        discriminator_optimizer.zero_grad()
        
        disc_loss.backward()
        discriminator_optimizer.step()
        
        ######
        disc_out = discriminator(gray_img, fake_img)
        adversarial_loss = bce_loss(disc_out, torch.ones(1, 1).to('cuda'))
        reconstruction_loss = l1_loss(fake_img, real_img)
        
        gen_loss = (weight_d * adversarial_loss + weight_g * reconstruction_loss)
        
        generator_optimizer.zero_grad()
        gen_loss.backward()
        generator_optimizer.step()
        
        disc_losses = np.append(disc_losses, disc_loss.detach().cpu())
        gen_losses = np.append(gen_losses, gen_loss.detach().cpu())
        adversarial_losses = np.append(adversarial_losses, adversarial_loss.detach().cpu())
        reconstruction_losses = np.append(reconstruction_losses, reconstruction_loss.detach().cpu())
    
    
    summary_writer.add_scalars(
        "Losses",
        {
            "D_Loss": disc_losses.mean(),
            "G_Loss": gen_losses.mean()
        },
        e
    )
    
    summary_writer.add_scalars(
        "Generator Losses",
        {
            "Adversarial Loss": adversarial_losses.mean(),
            "Reconstruction Loss": reconstruction_losses.mean()
        },
        e
    )
    
    summary_writer.add_scalars(
        "Discriminator Accuracies",
        {
            "Real Accuracy": real_accuracies.mean(),
            "Fake Accuracy": fake_accuracies.mean()
        },
        e
    )
    print("real accuracy : " + str(real_accuracies.mean()))
    print("fake accuracy : " + str(fake_accuracies.mean()))
    
    if (e + 1) % save_every == 0:
        torch.save(generator.state_dict(), 'weights_disc_0~1.pt')
        torch.save(discriminator.state_dict(),  'weights_disc_gen.pt')
    
    generator.eval()
    discriminator.eval()
    
    with torch.no_grad():
        losses = []
        disc_val_losses = []
        for batch in val_dataloader:
            label, img_input = batch
            
            label = label.to('cuda')
            img_input = img_input.to('cuda')

            gen_out = generator(img_input)
            disc_out = discriminator(img_input, gen_out)
            
            adversarial_loss = bce_loss(disc_out, torch.ones(1, 1).to('cuda'))
            reconstruction_loss = l1_loss(gen_out, label)
            loss = (weight_d * adversarial_loss + weight_g * reconstruction_loss)
            
            losses.append(loss)
            disc_val_losses.append(adversarial_loss)
            # Assume we're using a batch size of 1
            img_output = gen_out[0]
            # img_output = torch.permute(img_output, [1, 2, 0])
        
        mean_loss = sum(losses) / len(losses)
        mean_val_disc_loss = sum(disc_val_losses) / len(disc_val_losses)
        
        validation_reconstruction_losses = np.append(validation_reconstruction_losses, mean_loss.item())
        validation_adversarial_losses = np.append(validation_adversarial_losses, mean_val_disc_loss.item())
        summary_writer.add_scalar("Generator Validation Loss", mean_loss, e)
        summary_writer.add_scalar("Discriminator Validation Loss", mean_val_disc_loss, e)
        
        img = Image.fromarray(img_output.cpu().numpy(), mode="RGB")
        save_image(img_output, f'{store}epoch{e}_val_.jpg')
        save_image(label, f'{store}epoch{e}_val_original_.jpg')
    
    
    adversarial_loss_means = np.append(adversarial_loss_means, adversarial_losses.mean())
    disc_loss_means = np.append(disc_loss_means, disc_losses.mean())
    reconstruction_loss_means = np.append(reconstruction_loss_means, reconstruction_losses.mean())
    gen_loss_means = np.append(gen_loss_means, gen_losses.mean())
    
    plt.figure(figsize=(20, 15))
    plt.plot(cur_epochs, validation_adversarial_losses, 'k', label='Valid Adversarial Errors')
    plt.plot(cur_epochs, validation_reconstruction_losses, 'g', label='Valid Reconstruction Errors')
    plt.plot(cur_epochs, reconstruction_loss_means, 'b', label='Training Reconstruction Errors')
    plt.plot(cur_epochs, adversarial_loss_means, 'r', label='Adversarial Losses')
    plt.plot(cur_epochs, disc_loss_means, 'violet', label='Disc Losses')
    plt.plot(cur_epochs, gen_loss_means, 'limegreen', label='Generator Losses')
    
    plt.title('Validation and Test Errors')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    generator.train()
    discriminator.train()
# %%
