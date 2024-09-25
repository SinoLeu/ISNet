import torch
import torch.nn as nn
# from spikingjelly.activation_based import neuron, functional, surrogate, layer
from modules.neuron import MultiLIFNeuron
from spikingjelly.activation_based import neuron,layer

# class CGMNISTSpikingCNN(nn.Module):
#     def __init__(self,in_channel,num_classes=10):
#         super(CGMNISTSpikingCNN, self).__init__()
#         self.conv1 = nn.Sequential(
#              nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=1, padding=1),
#              MultiLIFNeuron()
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#             MultiLIFNeuron()
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#             MultiLIFNeuron()
#         )

#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#             MultiLIFNeuron()
#         )

#         self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 14 * 14, 512),
#             MultiLIFNeuron(),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.avg_pool(x)
#         x = self.fc(x)
#         return x

class CGMNISTSpikingCNN(nn.Module):
    def __init__(self, in_channel:int, channels: int, use_cupy=False):
        super().__init__()
        # self.T = T

        self.conv_fc = nn.Sequential(
        layer.Conv2d(6, channels, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels),
        MultiLIFNeuron(),
        layer.MaxPool2d(2, 2),  # 14 * 14

        layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels),
        MultiLIFNeuron(),
        layer.MaxPool2d(2, 2),  # 7 * 7

        layer.Flatten(),
        layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
        MultiLIFNeuron(),

        layer.Linear(channels * 4 * 4, 10, bias=False),
        neuron.LIFNode(),
        )


    def forward(self, x1: torch.Tensor,x2: torch.Tensor):
        # x.shape = [N, C, H, W]
        # x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        # x2 = gray_image
        x2 = x2.repeat(1,3,1,1)
        x = torch.cat((x1,x2),dim=1)
        
        fr = self.conv_fc(x)

        return fr

class CGMNISTSpikingCNN2(nn.Module):
    def __init__(self, in_channel:int, channels: int, use_cupy=False):
        super().__init__()
        # self.T = T

        self.conv_fc = nn.Sequential(
            nn.Conv2d(6, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            MultiLIFNeuron(),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            MultiLIFNeuron(),
            nn.MaxPool2d(2, 2),  # 7 * 7

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            MultiLIFNeuron(),
            nn.MaxPool2d(2, 2),  # 4 * 4

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            MultiLIFNeuron(),
            nn.MaxPool2d(2, 2),  # 2 * 2

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            MultiLIFNeuron(),
            nn.MaxPool2d(2, 2),  # 1 * 1

            nn.Flatten(),
            nn.Linear(channels * 1 * 1, channels * 4 * 4, bias=False),
            MultiLIFNeuron(),

            nn.Linear(channels * 4 * 4, 10, bias=False),
            neuron.LIFNode(),
        )


    def forward(self, x1: torch.Tensor,x2: torch.Tensor):
        # x.shape = [N, C, H, W]
        # x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        # x2 = gray_image
        x2 = x2.repeat(1,3,1,1)
        x = torch.cat((x1,x2),dim=1)
        
        fr = self.conv_fc(x)

        return fr
