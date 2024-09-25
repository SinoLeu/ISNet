import torch
import torch.nn as nn
import torch.nn.functional as F
# from spikingjelly.cext.neuron import MultiStepParametricLIFNode
# from spikingjelly.activation_based import neuron, functional, surrogate, layer
# from spikingjelly.clock_driven import layer
from modules.neuron import MultiLIFNeuron,ComplementaryLIFNeuron,VanillaNeuron

def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        # layer.SeqToANNContainer(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        # ),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
        ComplementaryLIFNeuron()
    )

def conv1x1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
        ComplementaryLIFNeuron()
    )


## fusion_f {Sum,element-wise product,Cross Attention,Concation,mlp}

class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f=None):
        super(SEWBlock, self).__init__()
        self.connect_f = connect_f
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            conv3x3(mid_channels, in_channels),
        )
    
    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        if self.connect_f == 'ADD':
            out = out + x  # use out-of-place add
        elif self.connect_f == 'AND':
            out = out * x  # use out-of-place multiply
        elif self.connect_f == 'IAND':
            out = x * (1. - out)  # ensure not in-place
        else:
            raise NotImplementedError(f"Connection function '{self.connect_f}' not implemented")

        return out


class WOInteractCell(nn.Module):
    def __init__(self, in_channels, mid_channels,connect_f=None):
        super(WOInteractCell, self).__init__()
        self.model1Block = SEWBlock(in_channels, mid_channels,connect_f)
        self.model2Block = SEWBlock(in_channels, mid_channels,connect_f)
        
    def forward(self,x1,x2):
        x1 = self.model1Block(x1)
        x2 = self.model1Block(x2)
        
        return x1,x2

class AttentionFusion(nn.Module):
    def __init__(self, channel_size):
        super(AttentionFusion, self).__init__()
        self.query_conv = nn.Conv2d(channel_size, channel_size // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channel_size, channel_size // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channel_size, channel_size, kernel_size=1)
        # self.softmax = nn.Softmax(dim=-1)  # Apply softmax to the last dimension
        self.active = VanillaNeuron()

    def forward(self, x1, x2):
        m_batchsize, C, height, width = x1.size()
        proj_query = self.query_conv(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x2).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)  # Batch matrix-matrix product
        attention = self.active(energy)
        proj_value = self.value_conv(x2).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        return out

## InteractSpikeNet_XS
class W_OInteractSpikeNet_XS(nn.Module):
    def __init__(self,pool_kernel_size=16, input_width=64, in_channels=[3, 1], num_classes=10, connect_f='ADD',fusion_f="concat"):
        super(W_OInteractSpikeNet_XS, self).__init__()
        self.fusion_f = fusion_f
        
        # Input channel initialization
        in1, in2 = in_channels
        out_channels = 64
        
        # Stage 1
        self.init_conv1_1 = conv3x3(in1, out_channels)
        self.init_conv1_2 = conv3x3(in2, out_channels)
        self.interact_cell1_1 = WOInteractCell(out_channels, out_channels, connect_f)
        
        # Stage 2
        self.conv2_1 = conv3x3(out_channels, 128)
        self.conv2_2 = conv3x3(out_channels, 128)
        self.interact_cell2_1 = WOInteractCell(128, 128, connect_f)
        
        # Stage 3
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(128, 256)
        self.interact_cell3_1 = WOInteractCell(256, 256, connect_f)
        
        
        # Pooling and final classification layer
        self.pool = nn.AvgPool2d(pool_kernel_size)
        
        # Adjust final_channels based on fusion method
        if self.fusion_f == "concat":
            final_channels = 256 * 2
        else:
            final_channels = 256
        if self.fusion_f == "att":
            self.attention_fusion = AttentionFusion(256)
            
        final_width = input_width // pool_kernel_size
        self.flatten = nn.Flatten()
        self.final_linear = nn.Linear(final_channels * final_width * final_width, num_classes)

    def forward(self, x1, x2):
        
        ## Stage 1
        x1 = self.init_conv1_1(x1)
        x2 = self.init_conv1_2(x2)
        x1, x2 = self.interact_cell1_1(x1, x2)        

        ## Stage 2 
        x1, x2 = self.conv2_1(x1), self.conv2_2(x2)
        x1, x2 = self.interact_cell2_1(x1, x2)
        
        ## Stage 3
        x1, x2 = self.conv3_1(x1), self.conv3_2(x2)
        x1, x2 = self.interact_cell3_1(x1, x2)
        
        # Apply the fusion function
        if self.fusion_f == "concat":
            x_fused = torch.cat((x1, x2), dim=1)
        elif self.fusion_f == "sum":
            x_fused = x1 + x2
        elif self.fusion_f == "ew":
            x_fused = x1 * x2
        elif self.fusion_f == "att":
            x_fused = self.attention_fusion(x1, x2)
        else:
            raise ValueError("Unsupported fusion function")
        
        # Concatenate feature maps, pool, and classify
        # x_concat = torch.cat((x1, x2), dim=1)
        pooled_x = self.pool(x_fused)
        return self.final_linear(self.flatten(pooled_x))


## InteractSpikeNet_S
class W_OInteractSpikeNet_S(nn.Module):
    def __init__(self,pool_kernel_size=16, input_width=64, in_channels=[3, 1], num_classes=10, connect_f='ADD',fusion_f="concat"):
        super(W_OInteractSpikeNet_S, self).__init__()
        self.fusion_f = fusion_f   
        # Input channel initialization
        in1, in2 = in_channels
        out_channels = 64
        
        # Stage 1
        self.init_conv1_1 = conv3x3(in1, out_channels)
        self.init_conv1_2 = conv3x3(in2, out_channels)
        self.interact_cell1_1 = WOInteractCell(out_channels, out_channels, connect_f)
        self.interact_cell1_2 = WOInteractCell(out_channels, out_channels, connect_f)
        
        # Stage 2
        self.conv2_1 = conv3x3(out_channels, 128)
        self.conv2_2 = conv3x3(out_channels, 128)
        self.interact_cell2_1 = WOInteractCell(128, 128, connect_f)
        self.interact_cell2_2 = WOInteractCell(128, 128, connect_f)
        
        # Stage 3
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(128, 256)
        self.interact_cell3_1 = WOInteractCell(256, 256, connect_f)
        self.interact_cell3_2 = WOInteractCell(256, 256, connect_f)
        
        # Pooling and final classification layer
        self.pool = nn.AvgPool2d(pool_kernel_size)
        # Adjust final_channels based on fusion method
        if self.fusion_f == "concat":
            final_channels = 256 * 2
        else:
            final_channels = 256
        if self.fusion_f == "att":
            self.attention_fusion = AttentionFusion(256)
            
        final_width = input_width // pool_kernel_size
        self.flatten = nn.Flatten()
        self.final_linear = nn.Linear(final_channels * final_width * final_width, num_classes)

    def forward(self, x1, x2):
        
        ## Stage 1
        x1 = self.init_conv1_1(x1)
        x2 = self.init_conv1_2(x2)
        x1, x2 = self.interact_cell1_1(x1, x2)   
        x1, x2 = self.interact_cell1_2(x1, x2)        

        ## Stage 2
        x1, x2 = self.conv2_1(x1), self.conv2_2(x2)
        x1, x2 = self.interact_cell2_1(x1, x2)
        x1, x2 = self.interact_cell2_2(x1, x2)
        
        ## Stage 3
        x1, x2 = self.conv3_1(x1), self.conv3_2(x2)
        x1, x2 = self.interact_cell3_1(x1, x2)
        x1, x2 = self.interact_cell3_2(x1, x2)
        
        
        # Apply the fusion function
        if self.fusion_f == "concat":
            x_fused = torch.cat((x1, x2), dim=1)
        elif self.fusion_f == "sum":
            x_fused = x1 + x2
        elif self.fusion_f == "ew":
            x_fused = x1 * x2
        elif self.fusion_f == "att":
            x_fused = self.attention_fusion(x1, x2)
        else:
            raise ValueError("Unsupported fusion function")
        pooled_x = self.pool(x_fused)
        return self.final_linear(self.flatten(pooled_x))
    
## InteractSpikeNet_L
class W_OInteractSpikeNet_L(nn.Module):
    def __init__(self, pool_kernel_size=16, input_width=64, in_channels=[3, 1], num_classes=10, connect_f='ADD',fusion_f="concat"):
        super(W_OInteractSpikeNet_L, self).__init__()
        self.fusion_f = fusion_f
        # Input channel initialization
        in1, in2 = in_channels
        out_channels = 64
        
        # Stage 1
        self.init_conv1_1 = conv3x3(in1, out_channels)
        self.init_conv1_2 = conv3x3(in2, out_channels)
        self.interact_cell1_1 = WOInteractCell(out_channels, out_channels, connect_f)
        self.interact_cell1_2 = WOInteractCell(out_channels, out_channels, connect_f)
        
        # Stage 2
        self.conv2_1 = conv3x3(out_channels, 128)
        self.conv2_2 = conv3x3(out_channels, 128)
        self.interact_cell2_1 = WOInteractCell(128, 128, connect_f)
        self.interact_cell2_2 = WOInteractCell(128, 128, connect_f)
        
        # Stage 3
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(128, 256)
        self.interact_cell3_1 = WOInteractCell(256, 256, connect_f)
        self.interact_cell3_2 = WOInteractCell(256, 256, connect_f)
        
        # Stage 4
        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(256, 512)
        self.interact_cell4_1 = WOInteractCell(512, 512, connect_f)
        
        # Pooling and final classification layer
        self.pool = nn.AvgPool2d(pool_kernel_size)
        # Adjust final_channels based on fusion method
        if self.fusion_f == "concat":
            final_channels = 512 * 2
        else:
            final_channels = 512
        if self.fusion_f == "att":
            self.attention_fusion = AttentionFusion(512)
            
        final_width = input_width // pool_kernel_size
        self.flatten = nn.Flatten()
        self.final_linear = nn.Linear(final_channels * final_width * final_width, num_classes)

    def forward(self, x1, x2):
        
        ## Stage 1
        x1 = self.init_conv1_1(x1)
        x2 = self.init_conv1_2(x2)
        x1, x2 = self.interact_cell1_1(x1, x2)   
        x1, x2 = self.interact_cell1_2(x1, x2)        

        ## Stage 2 
        x1, x2 = self.conv2_1(x1), self.conv2_2(x2)
        x1, x2 = self.interact_cell2_1(x1, x2)
        x1, x2 = self.interact_cell2_2(x1, x2)
        
        ## Stage 3
        x1, x2 = self.conv3_1(x1), self.conv3_2(x2)
        x1, x2 = self.interact_cell3_1(x1, x2)
        x1, x2 = self.interact_cell3_2(x1, x2)
        
        ## Stage 4
        x1, x2 = self.conv4_1(x1), self.conv4_2(x2)
        x1, x2 = self.interact_cell4_1(x1, x2)
        
        # Apply the fusion function
        if self.fusion_f == "concat":
            x_fused = torch.cat((x1, x2), dim=1)
        elif self.fusion_f == "sum":
            x_fused = x1 + x2
        elif self.fusion_f == "ew":
            x_fused = x1 * x2
        elif self.fusion_f == "att":
            x_fused = self.attention_fusion(x1, x2)
        else:
            raise ValueError("Unsupported fusion function")
        
        pooled_x = self.pool(x_fused)
        
        return self.final_linear(self.flatten(pooled_x))