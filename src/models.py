import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace = True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks = 9):
        super(Generator, self).__init__()

        ## Initial convolution block

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace = True)
        ]

        ## Downsampling

        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride = 2, padding = 1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace = True)]
            in_features = out_features
            out_features = in_features*2
        
        # Residual Blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # UpSampling
        out_features = in_features//2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride = 2, padding = 1, output_padding = 1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace = True)]
            in_features = out_features
            out_features = in_features//2

        ## Output Layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        ## A bunch of Convolution one after the other

        model = [
            nn.Conv2d(input_nc, 64, 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace = True)
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True)
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True)
        ]

        model += [
            nn.Conv2d(256, 512, 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        ]

        # FCN classification layer

        model += [
            nn.Conv2d(512, 1, 4, padding = 1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class Global_Discriminator(nn.Module):
    def __init__(self):
        super(Global_Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod((3,256,256))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
