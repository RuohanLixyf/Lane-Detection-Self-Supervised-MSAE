import torch
import config
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from utils import *
import operator
from config import args_setting


def generate_model(args):

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    assert args.model in [ 'UNet_ConvLSTM', 'SCNN_UNet_ConvLSTM', 'SCNN_UNet_Attention']
    if args.model == 'UNet_ConvLSTM':
        model = UNet_ConvLSTM(config.img_channel, config.class_num_train).to(device)
    elif args.model == 'SCNN_UNet_ConvLSTM':
        model = SCNN_UNet_ConvLSTM(config.img_channel, config.class_num_train).to(device)
    elif args.model == 'SCNN_UNet_Attention':
        model = SCNN_UNet_Attention(config.img_channel, config.class_num_train).to(device)
    return model


class UNet_ConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_ConvLSTM, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, config.class_num_train)
        self.convlstm = ConvLSTM(input_size=(8,16),
                                 input_dim=512,
                                 hidden_dim=[512, 512],
                                 kernel_size=(3,3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)
    def forward(self, x):
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.inc(item)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            data.append(x5.unsqueeze(0))
        data = torch.cat(data, dim=0)
        lstm, _ = self.convlstm(data)
        test = lstm[0][ -1,:, :, :, :]
        x = self.up1(test, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x, test
    
class SCNN_UNet_ConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SCNN_UNet_ConvLSTM, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4= up(128, 64)
        self.outc2 = outconv(64, config.class_num_train)
        self.max_pool = nn.MaxPool2d(2)
        self.convlstm = ConvLSTM(input_size=(8,16),
                                 input_dim=512,
                                 hidden_dim=[512, 512],
                                 kernel_size=(3,3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)
    # SCNN
        SCNN_kernel=9
        self.scnn_convs = nn.ModuleList()
        self.scnn_convs.add_module('SCNN_D', nn.Conv2d(64, 64, (1, SCNN_kernel), padding=(0, SCNN_kernel // 2), bias=False))
        self.scnn_convs.add_module('SCNN_U', nn.Conv2d(64, 64, (1, SCNN_kernel), padding=(0, SCNN_kernel // 2), bias=False))
        self.scnn_convs.add_module('SCNN_R', nn.Conv2d(64, 64, (SCNN_kernel, 1), padding=(SCNN_kernel // 2, 0), bias=False))
        self.scnn_convs.add_module('SCNN_L', nn.Conv2d(64, 64, (SCNN_kernel, 1), padding=(SCNN_kernel // 2, 0), bias=False))
        self.conv3 = nn.Conv2d(128, 2, 1)
    def scnn(self, x):
        type = ['V', 'V', 'H', 'H']
        direction = ['normal', 'reverse', 'normal', 'reverse']
        for ms_conv, v, r in zip(self.scnn_convs, type, direction):
            x = self.scnn_(x, ms_conv, v, r)
        return x
    def scnn_(self, x, conv, type='V', direction='normal'):
        assert type in ['V', 'H']
        assert direction in ['normal', 'reverse']
        B, C, H, W = x.size()

        if type == 'V':
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        elif type == 'H':
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if direction == 'reverse':
            slices = slices[::-1]
        # propagate the feature
        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if direction == 'reverse':
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def forward(self, x):
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.inc(item)
            x2 = self.down1(self.scnn(x1))
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            data.append(x5.unsqueeze(0))
        data = torch.cat(data, dim=0)
        lstm, _ = self.convlstm(data)
        test = lstm[0][ -1,:, :, :, :]
        x = self.up1(test, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc2(x)
        return x, test

class SCNN_UNet_Attention(nn.Module):
    def __init__(self, n_channels, n_classes):
          super(SCNN_UNet_Attention, self).__init__()
          self.inc = inconv(n_channels, 64)
          self.down1 = downfirst(64, 128)
          self.down2 = down(128, 256)
          self.down3 = down(256, 512)
          self.down4 = down(512, 512)
          self.up1 = up(1024, 256)
          self.up2 = up(512, 128)
          self.up3 = up(256, 64)
          self.up4 = up(128, 64)
          self.inconv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(1, 1))
          self.outconv = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 1))
          self.outc1 = outconv(64, config.class_num_train)
          self.attention_module = AttentionModule(input_size=128, hidden_size=128)
          # SCNN Part
          SCNN_kernel=9
          self.scnn_convs = nn.ModuleList()
          self.scnn_convs.add_module('SCNN_D', nn.Conv2d(64, 64, (1, SCNN_kernel), padding=(0, SCNN_kernel // 2), bias=False))
          self.scnn_convs.add_module('SCNN_U', nn.Conv2d(64, 64, (1, SCNN_kernel), padding=(0, SCNN_kernel // 2), bias=False))
          self.scnn_convs.add_module('SCNN_R', nn.Conv2d(64, 64, (SCNN_kernel, 1), padding=(SCNN_kernel // 2, 0), bias=False))
          self.scnn_convs.add_module('SCNN_L', nn.Conv2d(64, 64, (SCNN_kernel, 1), padding=(SCNN_kernel // 2, 0), bias=False))
    
    def scnn(self, x):
          type = ['V', 'V', 'H', 'H']
          direction = ['normal', 'reverse', 'normal', 'reverse']
          for ms_conv, v, r in zip(self.scnn_convs, type, direction):
              x = self.scnn_(x, ms_conv, v, r)
          return x
    def scnn_(self, x, conv, type='V', direction='normal'):
          assert type in ['V', 'H']
          assert direction in ['normal', 'reverse']
          B, C, H, W = x.size()
          # turn the feature maps into slices
          if type == 'V':
              slices = [x[:, :, i:(i + 1), :] for i in range(H)]
              dim = 2
          elif type == 'H':
              slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
              dim = 3
          if direction == 'reverse':
              slices = slices[::-1]
          # propagate the feature
          out = [slices[0]]
          for i in range(1, len(slices)):
              out.append(slices[i] + F.relu(conv(out[i - 1])))
          if direction == 'reverse':
              out = out[::-1]
          return torch.cat(out, dim=dim)

    def forward(self, x):
          x = torch.unbind(x, dim=1)
          data = []
          for item in x:
              x1 = self.inc(item)
              x2 = self.down1(self.scnn(x1))
              x3 = self.down2(x2)
              x4 = self.down3(x3)
              x5 = self.down4(x4)
              x6 = self.inconv(x5)
              data.append(x6.unsqueeze(0))
          data = torch.cat(data, dim=0)
          data = torch.flatten(data, start_dim=2)
          test, _ = self.attention_module(data)
          output_tensor = test.permute(1, 0, 2)
          output_tensor_new = output_tensor.reshape(output_tensor.shape[0], output_tensor.shape[1], 8, 16)
          output = self.outconv(output_tensor_new)
          x = self.up1(output, x4)
          x = self.up2(x, x3)
          x = self.up3(x, x2)
          x = self.up4(x, x1)
          x = self.outc1(x)
          return x, test


