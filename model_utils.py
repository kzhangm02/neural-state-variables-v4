import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def conv2d_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def ffn_bn(in_dim, out_dim):
    ffnlayer = torch.nn.Sequential(
        torch.nn.Linear(in_dim, out_dim),
        torch.nn.BatchNorm1d(out_dim),
    )
    return ffnlayer

def ffn_bn_relu(in_dim, out_dim):
    ffnlayer = torch.nn.Sequential(
        torch.nn.Linear(in_dim, out_dim),
        torch.nn.BatchNorm1d(out_dim),
        torch.nn.ReLU()
    )
    return ffnlayer

def deconv_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def deconv_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer

def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class EncoderDecoder64x1x1(torch.nn.Module):
    def __init__(self, in_channels):
        super(EncoderDecoder64x1x1,self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(in_channels,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=2),
            conv2d_bn_relu(64,64,3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )
        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )
        self.conv_stack6 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )
        self.conv_stack7 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )
        self.conv_stack8 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,(3,4),stride=(1,2)),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_ffn = ffn_bn(64, 128)
        self.deconv_ffn = ffn_bn_relu(64, 64)
        
        self.deconv_8 = deconv_bn_relu(64,64,(3,4),stride=(1,2))
        self.deconv_7 = deconv_bn_relu(67,64,4,stride=2)
        self.deconv_6 = deconv_bn_relu(67,64,4,stride=2)
        self.deconv_5 = deconv_bn_relu(67,64,4,stride=2)
        self.deconv_4 = deconv_bn_relu(67,64,4,stride=2)
        self.deconv_3 = deconv_bn_relu(67,32,4,stride=2)
        self.deconv_2 = deconv_bn_relu(35,16,4,stride=2)
        self.deconv_1 = deconv_sigmoid(19,3,4,stride=2)

        self.predict_8 = torch.nn.Conv2d(64,3,3,stride=1,padding=1)
        self.predict_7 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_6 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_5 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_4 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_3 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_2 = torch.nn.Conv2d(35,3,3,stride=1,padding=1)

        self.up_sample_8 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,(3,4),stride=(1,2),padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        conv6_out = self.conv_stack6(conv5_out)
        conv7_out = self.conv_stack7(conv6_out)
        conv8_out = self.conv_stack8(conv7_out)
        conv8_out = torch.reshape(conv8_out, (-1, 64))
        ffn_out = self.conv_ffn(conv8_out)
        return ffn_out

    def decoder(self, x):
        deconv_ffn_out = self.deconv_ffn(x).reshape((-1, 64, 1, 1))
        deconv8_out = self.deconv_8(deconv_ffn_out)
        predict_8_out = self.up_sample_8(self.predict_8(deconv_ffn_out))

        concat_7 = torch.cat([deconv8_out, predict_8_out], dim=1)
        deconv7_out = self.deconv_7(concat_7)
        predict_7_out = self.up_sample_7(self.predict_7(concat_7))

        concat_6 = torch.cat([deconv7_out,predict_7_out],dim=1)
        deconv6_out = self.deconv_6(concat_6)
        predict_6_out = self.up_sample_6(self.predict_6(concat_6))
        
        concat_5 = torch.cat([deconv6_out,predict_6_out],dim=1)
        deconv5_out = self.deconv_5(concat_5)
        predict_5_out = self.up_sample_5(self.predict_5(concat_5))

        concat_4 = torch.cat([deconv5_out,predict_5_out],dim=1)
        deconv4_out = self.deconv_4(concat_4)
        predict_4_out = self.up_sample_4(self.predict_4(concat_4))

        concat_3 = torch.cat([deconv4_out,predict_4_out],dim=1)
        deconv3_out = self.deconv_3(concat_3)
        predict_3_out = self.up_sample_3(self.predict_3(concat_3))

        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        predict_out = self.deconv_1(concat1)
        return predict_out

    def forward(self, x, reconstructed_latent):
        if reconstructed_latent is None:
            distributions = self.encoder(x)
            mu = distributions[:, :64]
            logvar = distributions[:, 64:]
            z = reparametrize(mu, logvar)
            rx = self.decoder(z).view(x.size())
            return rx, z, mu, logvar
        else:
            distributions = self.encoder(x)
            mu = distributions[:, :64]
            logvar = distributions[:, 64:]
            z = reparametrize(mu, logvar)
            rx = self.decoder(reconstructed_latent).view(x.size())
            return rx, z, mu, logvar


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)
    

class EncoderDecoderDynamicsNetwork(torch.nn.Module):
    def __init__(self, in_channels):
        super(EncoderDecoderDynamicsNetwork,self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 256)
        self.layer3 = SirenLayer(256, 128)
        self.layer4 = SirenLayer(128, in_channels, is_last=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class RefineCircularMotionModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineCircularMotionModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 2)
        self.layer5 = SirenLayer(2, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)
    
    def encoder(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        return latent
    
    def decoder(self, latent):
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent


class CircularMotionDynamicsModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(CircularMotionDynamicsModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 32, is_first=True)
        self.layer2 = SirenLayer(32, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class RefineReactionDiffusionModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineReactionDiffusionModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 2)
        self.layer5 = SirenLayer(2, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent


class RefineSinglePendulumModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineSinglePendulumModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 2)
        self.layer5 = SirenLayer(2, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)
    
    def encoder(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        return latent
    
    def decoder(self, latent):
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent


class SinglePendulumDynamicsModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(SinglePendulumDynamicsModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 32, is_first=True)
        self.layer2 = SirenLayer(32, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class RefineDoublePendulumModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineDoublePendulumModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 4)
        self.layer5 = SirenLayer(4, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def encoder(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        return latent
    
    def decoder(self, latent):
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent
    

class DoublePendulumDynamicsModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(DoublePendulumDynamicsModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 32, is_first=True)
        self.layer2 = SirenLayer(32, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class RefineElasticPendulumModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineElasticPendulumModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 6)
        self.layer5 = SirenLayer(6, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent


class RefineLavaLampModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineLavaLampModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 4)
        self.layer5 = SirenLayer(4, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent