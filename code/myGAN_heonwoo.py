import torch
import torch.nn as nn 

class GaussianNoise(nn.Module):                   # Try noise just for real or just for fake images.
    def __init__(self, std=0.1, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(16),#
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),#
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),#
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.InstanceNorm2d(32)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),#
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),#
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.InstanceNorm2d(64)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),#
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),#
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.InstanceNorm2d(128)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),#
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),#
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.InstanceNorm2d(256)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),#
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),#
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.InstanceNorm2d(512)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),#
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.InstanceNorm2d(512)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),#
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(512)
        )

        self.conv9_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(512),#
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(512)
        )

        self.conv9_2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),#
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(512)
        )

        self.conv10_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256),#
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(256)
        )

        self.conv10_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),#
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(256)
        )

        self.conv11_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),#
            nn.LeakyReLU(),
        )

        self.conv11_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),#
            nn.LeakyReLU(),
        )

        self.conv12_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),#
            nn.LeakyReLU(),
        )

        self.conv12_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),#
            nn.LeakyReLU(),
        )

        self.conv13_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),#
            nn.LeakyReLU(),
        )

        self.conv13_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),#
            nn.LeakyReLU(),
        )

        self.conv14_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(16),#
            nn.LeakyReLU(),
        )

        self.conv14_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(16),#
            nn.LeakyReLU(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, padding=0),
            #nn.ReLU(),
            nn.Tanh()
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9_1 = self.conv9_1(x8)
        x9_2 = torch.cat((x9_1, x6), dim=1)
        x9 = self.conv9_2(x9_2)
        x10_1 = self.conv10_1(x9)
        x10_2 = torch.cat((x10_1, x5), dim=1)
        x10 = self.conv10_2(x10_2)
        x11_1 = self.conv11_1(x10)
        x11_2 = torch.cat((x11_1, x4), dim=1)
        x11 = self.conv11_2(x11_2)
        x12_1 = self.conv12_1(x11)
        x12_2 = torch.cat((x12_1, x3), dim=1)
        x12 = self.conv12_2(x12_2)
        x13_1 = self.conv13_1(x12)
        x13_2 = torch.cat((x13_1, x2), dim=1)
        x13 = self.conv13_2(x13_2)
        x14_1 = self.conv14_1(x13)
        x14_2 = torch.cat((x14_1, x1), dim=1)
        x14 = self.conv14_2(x14_2)
        out = self.out(x14)
        
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        
        self.conv1_i = nn.Sequential(
            GaussianNoise(std=0.1, decay_rate=0), 
            #nn.InstanceNorm2d(1),
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(32)
        )

        self.conv1_o = nn.Sequential(
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.InstanceNorm2d(3),
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(32)
        )

        self.conv2_i = nn.Sequential(
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            
            #nn.InstanceNorm2d(64)
        )

        self.conv2_o = nn.Sequential(
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(64)
        )

        self.conv3_i = nn.Sequential(
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64), # 추가
            nn.LeakyReLU()
        )

        self.conv3_o = nn.Sequential(
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64), #추가
            nn.LeakyReLU()
        )

        self.conv4 = nn.Sequential(
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
           # nn.InstanceNorm2d(128),
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
          #  nn.InstanceNorm2d(128),
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
            #nn.InstanceNorm2d(128),
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
           # nn.InstanceNorm2d(256),
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
           # nn.InstanceNorm2d(256)
        )

        self.fc = nn.Sequential(
            GaussianNoise(std=0.1, decay_rate=0), 
            nn.Flatten(),
            nn.Linear(16384, 4000),  # Adjusted input dimension
            nn.LeakyReLU(),
            nn.Linear(4000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 250),
            nn.LeakyReLU(),
            nn.Linear(250, 1),
            nn.Sigmoid()
        )

    def forward(self, generator_input, generator_output):
        x1 = self.conv1_i(generator_input)
        x2 = self.conv1_o(generator_output)
        x1 = self.conv2_i(x1)
        x2 = self.conv2_o(x2)
        x1 = self.conv3_i(x1)
        x2 = self.conv3_o(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.conv4(x)
        output = self.fc(x)

        return output