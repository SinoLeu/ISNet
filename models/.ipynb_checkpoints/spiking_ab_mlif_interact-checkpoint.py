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
        # layer.SeqToANNContainer(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        # ),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
       
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

## w/o mlif
class InteractCell(nn.Module):
    def __init__(self, in_channels, mid_channels,connect_f=None):
        super(InteractCell, self).__init__()
        self.model1Block = SEWBlock(in_channels, mid_channels,connect_f)
        self.model2Block = SEWBlock(in_channels, mid_channels,connect_f)
        self.w_1 = conv1x1(in_channels,in_channels)
        self.w_2 = conv1x1(in_channels,in_channels)
        self.mlif = MultiLIFNeuron()
    def forward(self,x1,x2):
        x1 = self.model1Block(x1)
        x2 = self.model1Block(x2)
        proj_x1 = self.w_1(x1)
        proj_x2 = self.w_2(x2)
        # combine = torch.cat((proj_x1,proj_x2),dim=1)
        # spike = self.mlif(combine)
        # new_x1,new_x2 = torch.chunk(spike,chunks=2,dim=1)
        return x1+proj_x1,x2+proj_x2

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

class MLPFusion(nn.Module):
    def __init__(self, channel_size, hidden_size):
        super(MLPFusion, self).__init__()
        # Define two separate pathways for each input
        self.fc1_x1 = nn.Linear(channel_size, hidden_size)
        self.fc1_x2 = nn.Linear(channel_size, hidden_size)

        self.lif1 = ComplementaryLIFNeuron()
        self.lfi2 = ComplementaryLIFNeuron()
        # A fusion layer that combines the pathways
        self.fc2_fusion = nn.Linear(hidden_size * 2, channel_size)

    def forward(self, x1, x2):
        m_batchsize, C, height, width = x1.size()

        # Flatten the spatial dimensions for MLP processing
        x1_flat = x1.view(m_batchsize, C, -1).permute(0, 2, 1)  # Shape: [m_batchsize, height*width, C]
        x2_flat = x2.view(m_batchsize, C, -1).permute(0, 2, 1)  # Shape: [m_batchsize, height*width, C]

        # Apply the first layer MLPs
        x1_transformed = self.lif1(self.fc1_x1(x1_flat))  # Shape: [m_batchsize, height*width, hidden_size]
        x2_transformed = self.lfi2(self.fc1_x2(x2_flat))  # Shape: [m_batchsize, height*width, hidden_size]

        # Concatenate along the feature dimension
        fusion_input = torch.cat([x1_transformed, x2_transformed], dim=-1)  # Shape: [m_batchsize, height*width, hidden_size*2]

        # Fusion layer
        fused_features = self.fc2_fusion(fusion_input)  # Shape: [m_batchsize, height*width, channel_size]

        # Reshape to original spatial dimensions
        out = fused_features.permute(0, 2, 1).view(m_batchsize, C, height, width)

        return out

## InteractSpikeNet_XS
class InteractSpikeNet_XS(nn.Module):
    def __init__(self, pool_kernel_size=16, input_width=64, in_channels=[3, 1], num_classes=10, connect_f='ADD', fusion_f="concat",is_training=True):
        super(InteractSpikeNet_XS, self).__init__()
        self.fusion_f = fusion_f
        self.is_training = is_training
        # Input channel initialization
        in1, in2 = in_channels
        out_channels = 64
        
        # Stage 1
        self.init_conv1_1 = conv3x3(in1, out_channels)
        self.init_conv1_2 = conv3x3(in2, out_channels)
        self.interact_cell1_1 = InteractCell(out_channels, out_channels, connect_f)
        self.pool1 = nn.AvgPool2d(2, stride=2)  # Add avg pool here with stride 2
        
        # Stage 2
        self.conv2_1 = conv3x3(out_channels, 128)
        self.conv2_2 = conv3x3(out_channels, 128)
        self.interact_cell2_1 = InteractCell(128, 128, connect_f)
        self.pool2 = nn.AvgPool2d(2, stride=2)  # Add avg pool here with stride 2
        
        # Stage 3
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(128, 256)
        self.interact_cell3_1 = InteractCell(256, 256, connect_f)
        self.pool3 = nn.AvgPool2d(2, stride=2)  # Add avg pool here with stride 2
        
        # Final pooling and classification layer
        self.pool = nn.AvgPool2d(2, stride=2)  # Adjusted for final reduction
        
        # Adjust final_channels based on fusion method
        if self.fusion_f == "concat":
            final_channels = 256 * 2
        else:
            final_channels = 256
        singal_channels = 256
        if self.fusion_f == "att":
            self.attention_fusion = AttentionFusion(256)
        if self.fusion_f == 'mlp':
            self.mlp_fusion = MLPFusion(256, 256)
        
        final_width = input_width // (2 * 2 * 2 * 2)  # Updated final width calculation
        self.flatten = nn.Flatten()
        self.final_linear = nn.Linear(final_channels * final_width * final_width, num_classes)
        if is_training:
            self.unimodal_linear1 = nn.Linear(singal_channels * final_width * final_width, num_classes)
            self.unimodal_linear2 = nn.Linear(singal_channels * final_width * final_width, num_classes)
    def forward(self, x1, x2):
        ## Stage 1
        x1 = self.init_conv1_1(x1)
        x2 = self.init_conv1_2(x2)
        x1, x2 = self.interact_cell1_1(x1, x2)        
        x1, x2 = self.pool1(x1), self.pool1(x2)  # Apply avg pool after stage 1
        
        ## Stage 2 
        x1, x2 = self.conv2_1(x1), self.conv2_2(x2)
        x1, x2 = self.interact_cell2_1(x1, x2)
        x1, x2 = self.pool2(x1), self.pool2(x2)  # Apply avg pool after stage 2
        
        ## Stage 3
        x1, x2 = self.conv3_1(x1), self.conv3_2(x2)
        x1, x2 = self.interact_cell3_1(x1, x2)
        x1, x2 = self.pool3(x1), self.pool3(x2)  # Apply avg pool after stage 3
        
        # Apply the fusion function
        if self.fusion_f == "concat":
            x_fused = torch.cat((x1, x2), dim=1)
        elif self.fusion_f == "sum":
            x_fused = x1 + x2
        elif self.fusion_f == "ew":
            x_fused = x1 * x2
        elif self.fusion_f == "att":
            x_fused = self.attention_fusion(x1, x2)
        elif self.fusion_f == 'mlp':
            x_fused = self.mlp_fusion(x1, x2)
        else:
            raise ValueError("Unsupported fusion function")
        
        pooled_x = self.pool(x_fused)
        pooled_x1 = self.pool(x1)
        pooled_x2 = self.pool(x2)
        
        ## 
        if self.is_training:
            return self.final_linear(self.flatten(pooled_x)),self.unimodal_linear1(self.flatten(pooled_x1)),self.unimodal_linear2(self.flatten(pooled_x2))
        else:
            return self.final_linear(self.flatten(pooled_x))


## InteractSpikeNet_S
class InteractSpikeNet_S(nn.Module):
    def __init__(self, pool_kernel_size=8, input_width=64, in_channels=[3, 1], num_classes=10, connect_f='ADD', fusion_f="concat",is_training=True):
        super(InteractSpikeNet_S, self).__init__()
        self.fusion_f = fusion_f
        self.is_training = is_training
        # Input channel initialization
        in1, in2 = in_channels
        out_channels = 64
        
        # Stage 1
        self.init_conv1_1 = conv3x3(in1, out_channels)
        self.init_conv1_2 = conv3x3(in2, out_channels)
        self.interact_cell1_1 = InteractCell(out_channels, out_channels, connect_f)
        self.interact_cell1_2 = InteractCell(out_channels, out_channels, connect_f)
        self.pool1 = nn.AvgPool2d(2, stride=2)  # Add avg pool here with stride 2
        
        # Stage 2
        self.conv2_1 = conv3x3(out_channels, 128)
        self.conv2_2 = conv3x3(out_channels, 128)
        self.interact_cell2_1 = InteractCell(128, 128, connect_f)
        self.interact_cell2_2 = InteractCell(128, 128, connect_f)
        self.pool2 = nn.AvgPool2d(2, stride=2)  # Add avg pool here with stride 2
        
        # Stage 3
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(128, 256)
        self.interact_cell3_1 = InteractCell(256, 256, connect_f)
        self.interact_cell3_2 = InteractCell(256, 256, connect_f)
        self.pool3 = nn.AvgPool2d(2, stride=2)  # Add avg pool here with stride 2
        
        # Pooling and final classification layer
        self.pool = nn.AvgPool2d(2, stride=2)
        # Adjust final_channels based on fusion method
        if self.fusion_f == "concat":
            final_channels = 256 * 2
        else:
            final_channels = 256
        if self.fusion_f == "att":
            self.attention_fusion = AttentionFusion(256)
        if self.fusion_f == 'mlp':
            self.mlp_fusion = MLPFusion(256,256)
            
        final_width = input_width // (2*2*2*2)
        singal_channels = 256
        # print(final_channels * final_width * final_width)
        self.flatten = nn.Flatten()
        self.final_linear = nn.Linear(final_channels * final_width * final_width, num_classes)
        if is_training:
            self.unimodal_linear1 = nn.Linear(singal_channels * final_width * final_width, num_classes)
            self.unimodal_linear2 = nn.Linear(singal_channels * final_width * final_width, num_classes)
    def forward(self, x1, x2):
        
        ## Stage 1
        x1 = self.init_conv1_1(x1)
        x2 = self.init_conv1_2(x2)
        x1, x2 = self.interact_cell1_1(x1, x2)   
        x1, x2 = self.interact_cell1_2(x1, x2)
        x1, x2 = self.pool1(x1), self.pool1(x2)  # Apply avg pool after stage 1

        ## Stage 2
        x1, x2 = self.conv2_1(x1), self.conv2_2(x2)
        x1, x2 = self.interact_cell2_1(x1, x2)
        x1, x2 = self.interact_cell2_2(x1, x2)
        x1, x2 = self.pool2(x1), self.pool2(x2)  # Apply avg pool after stage 2

        ## Stage 3
        x1, x2 = self.conv3_1(x1), self.conv3_2(x2)
        x1, x2 = self.interact_cell3_1(x1, x2)
        x1, x2 = self.interact_cell3_2(x1, x2)
        x1, x2 = self.pool3(x1), self.pool3(x2)  # Apply avg pool after stage 3
        
        # Apply the fusion function
        if self.fusion_f == "concat":
            x_fused = torch.cat((x1, x2), dim=1)
        elif self.fusion_f == "sum":
            x_fused = x1 + x2
        elif self.fusion_f == "ew":
            x_fused = x1 * x2
        elif self.fusion_f == "att":
            x_fused = self.attention_fusion(x1, x2)
        elif self.fusion_f == 'mlp':
            x_fused = self.mlp_fusion(x1,x2)
        else:
            raise ValueError("Unsupported fusion function")
        ## 
        pooled_x = self.pool(x_fused)
        pooled_x1 = self.pool(x1)
        pooled_x2 = self.pool(x2)
        # print(pooled_x.shape)
        if self.is_training:
            return self.final_linear(self.flatten(pooled_x)),self.unimodal_linear1(self.flatten(pooled_x1)),self.unimodal_linear2(self.flatten(pooled_x2))
        else:
            return self.final_linear(self.flatten(pooled_x))

    
## InteractSpikeNet_L
class InteractSpikeNet_L(nn.Module):
    def __init__(self, pool_kernel_size=16, input_width=64, in_channels=[3, 1], num_classes=10, connect_f='ADD', fusion_f="concat",is_training=True):
        super(InteractSpikeNet_L, self).__init__()
        self.fusion_f = fusion_f
        self.is_training = is_training
        # Input channel initialization
        in1, in2 = in_channels
        out_channels = 64
        
        # Stage 1
        self.init_conv1_1 = conv3x3(in1, out_channels)
        self.init_conv1_2 = conv3x3(in2, out_channels)
        self.interact_cell1_1 = InteractCell(out_channels, out_channels, connect_f)
        self.interact_cell1_2 = InteractCell(out_channels, out_channels, connect_f)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Stage 2
        self.conv2_1 = conv3x3(out_channels, 128)
        self.conv2_2 = conv3x3(out_channels, 128)
        self.interact_cell2_1 = InteractCell(128, 128, connect_f)
        self.interact_cell2_2 = InteractCell(128, 128, connect_f)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.dropout2 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Stage 3
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(128, 256)
        self.interact_cell3_1 = InteractCell(256, 256, connect_f)
        self.interact_cell3_2 = InteractCell(256, 256, connect_f)
        self.pool3 = nn.AvgPool2d(2, stride=2)
        self.dropout3 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Stage 4
        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(256, 512)
        self.interact_cell4_1 = InteractCell(512, 512, connect_f)
        self.pool4 = nn.AvgPool2d(2, stride=2)
        self.dropout4 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Final pooling and classification layer
        self.pool = nn.AvgPool2d(4, stride=4)
        
        # Adjust final_channels based on fusion method
        if self.fusion_f == "concat":
            final_channels = 512 * 2
        else:
            final_channels = 512
        
        if self.fusion_f == "att":
            self.attention_fusion = AttentionFusion(512)
        if self.fusion_f == 'mlp':
            self.mlp_fusion = MLPFusion(512, 512)
        singal_channels = 512
        final_width = input_width // (2 * 2 * 2 * 2 * 4)
        self.flatten = nn.Flatten()
        self.dropout_final = nn.Dropout(p=0.3)  # Add dropout before the final linear layer
        self.final_linear = nn.Linear(final_channels * final_width * final_width, num_classes)
        if is_training:
            self.unimodal_linear1 = nn.Linear(singal_channels * final_width * final_width, num_classes)
            self.unimodal_linear2 = nn.Linear(singal_channels * final_width * final_width, num_classes)
    def forward(self, x1, x2):
        
        ## Stage 1
        x1 = self.init_conv1_1(x1)
        x2 = self.init_conv1_2(x2)
        x1, x2 = self.interact_cell1_1(x1, x2)   
        x1, x2 = self.interact_cell1_2(x1, x2)        
        x1, x2 = self.pool1(x1), self.pool1(x2)
        x1, x2 = self.dropout1(x1), self.dropout1(x2)  # Apply dropout after pool1

        ## Stage 2 
        x1, x2 = self.conv2_1(x1), self.conv2_2(x2)
        x1, x2 = self.interact_cell2_1(x1, x2)
        x1, x2 = self.interact_cell2_2(x1, x2)
        x1, x2 = self.pool2(x1), self.pool2(x2)
        x1, x2 = self.dropout2(x1), self.dropout2(x2)  # Apply dropout after pool2
        
        ## Stage 3
        x1, x2 = self.conv3_1(x1), self.conv3_2(x2)
        x1, x2 = self.interact_cell3_1(x1, x2)
        x1, x2 = self.interact_cell3_2(x1, x2)
        x1, x2 = self.pool3(x1), self.pool3(x2)
        x1, x2 = self.dropout3(x1), self.dropout3(x2)  # Apply dropout after pool3
        
        ## Stage 4
        x1, x2 = self.conv4_1(x1), self.conv4_2(x2)
        x1, x2 = self.interact_cell4_1(x1, x2)
        x1, x2 = self.pool4(x1), self.pool4(x2)
        x1, x2 = self.dropout4(x1), self.dropout4(x2)  # Apply dropout after pool4
        
        # Apply the fusion function
        if self.fusion_f == "concat":
            x_fused = torch.cat((x1, x2), dim=1)
        elif self.fusion_f == "sum":
            x_fused = x1 + x2
        elif self.fusion_f == "ew":
            x_fused = x1 * x2
        elif self.fusion_f == "att":
            x_fused = self.attention_fusion(x1, x2)
        elif self.fusion_f == 'mlp':
            x_fused = self.mlp_fusion(x1, x2)
        else:
            raise ValueError("Unsupported fusion function")
        
        # Final reduction and classification
        pooled_x = self.pool(x_fused)
        pooled_x = self.dropout_final(pooled_x)  # Apply dropout before final linear layer
        pooled_x1 =  self.dropout_final(self.pool(x1))
        pooled_x2 = self.dropout_final(self.pool(x2))
        ## 
        if self.is_training:
            return self.final_linear(self.flatten(pooled_x)),self.unimodal_linear1(self.flatten(pooled_x1)),self.unimodal_linear2(self.flatten(pooled_x2))
        else:
            return self.final_linear(self.flatten(pooled_x))

        
## InteractSpikeNet_XL
class InteractSpikeNet_XL(nn.Module):
    def __init__(self, pool_kernel_size=16, input_width=64, in_channels=[3, 1], num_classes=10, connect_f='ADD', fusion_f="concat",is_training=True):
        super(InteractSpikeNet_XL, self).__init__()
        self.fusion_f = fusion_f
        self.is_training = is_training
        # Input channel initialization
        in1, in2 = in_channels
        out_channels = 64
        
        # Stage 1
        self.init_conv1_1 = conv3x3(in1, out_channels)
        self.init_conv1_2 = conv3x3(in2, out_channels)
        self.interact_cell1_1 = InteractCell(out_channels, out_channels, connect_f)
        self.interact_cell1_2 = InteractCell(out_channels, out_channels, connect_f)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Stage 2
        self.conv2_1 = conv3x3(out_channels, 128)
        self.conv2_2 = conv3x3(out_channels, 128)
        self.interact_cell2_1 = InteractCell(128, 128, connect_f)
        self.interact_cell2_2 = InteractCell(128, 128, connect_f)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.dropout2 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Stage 3
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(128, 256)
        self.interact_cell3_1 = InteractCell(256, 256, connect_f)
        self.interact_cell3_2 = InteractCell(256, 256, connect_f)
        self.pool3 = nn.AvgPool2d(2, stride=2)
        self.dropout3 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Stage 4
        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(256, 512)
        self.interact_cell4_1 = InteractCell(512, 512, connect_f)
        self.interact_cell4_2 = InteractCell(512, 512, connect_f)
        self.pool4 = nn.AvgPool2d(2, stride=2)
        self.dropout4 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Final pooling and classification layer
        self.pool = nn.AvgPool2d(4, stride=4)
        
        # Adjust final_channels based on fusion method
        if self.fusion_f == "concat":
            final_channels = 512 * 2
        else:
            final_channels = 512
        
        if self.fusion_f == "att":
            self.attention_fusion = AttentionFusion(512)
        if self.fusion_f == 'mlp':
            self.mlp_fusion = MLPFusion(512, 512)
        singal_channels = 512
        final_width = input_width // (2 * 2 * 2 * 2 * 4)
        self.flatten = nn.Flatten()
        self.dropout_final = nn.Dropout(p=0.3)  # Add dropout before the final linear layer
        self.final_linear = nn.Linear(final_channels * final_width * final_width, num_classes)
        if is_training:
            self.unimodal_linear1 = nn.Linear(singal_channels * final_width * final_width, num_classes)
            self.unimodal_linear2 = nn.Linear(singal_channels * final_width * final_width, num_classes)
    def forward(self, x1, x2):
        
        ## Stage 1
        x1 = self.init_conv1_1(x1)
        x2 = self.init_conv1_2(x2)
        x1, x2 = self.interact_cell1_1(x1, x2)   
        x1, x2 = self.interact_cell1_2(x1, x2)        
        x1, x2 = self.pool1(x1), self.pool1(x2)
        x1, x2 = self.dropout1(x1), self.dropout1(x2)  # Apply dropout after pool1

        ## Stage 2 
        x1, x2 = self.conv2_1(x1), self.conv2_2(x2)
        x1, x2 = self.interact_cell2_1(x1, x2)
        x1, x2 = self.interact_cell2_2(x1, x2)
        x1, x2 = self.pool2(x1), self.pool2(x2)
        x1, x2 = self.dropout2(x1), self.dropout2(x2)  # Apply dropout after pool2
        
        ## Stage 3
        x1, x2 = self.conv3_1(x1), self.conv3_2(x2)
        x1, x2 = self.interact_cell3_1(x1, x2)
        x1, x2 = self.interact_cell3_2(x1, x2)
        x1, x2 = self.pool3(x1), self.pool3(x2)
        x1, x2 = self.dropout3(x1), self.dropout3(x2)  # Apply dropout after pool3
        
        ## Stage 4
        x1, x2 = self.conv4_1(x1), self.conv4_2(x2)
        x1, x2 = self.interact_cell4_1(x1, x2)
        x1, x2 = self.interact_cell4_2(x1, x2)
        x1, x2 = self.pool4(x1), self.pool4(x2)
        x1, x2 = self.dropout4(x1), self.dropout4(x2)  # Apply dropout after pool4
        
        # Apply the fusion function
        if self.fusion_f == "concat":
            x_fused = torch.cat((x1, x2), dim=1)
        elif self.fusion_f == "sum":
            x_fused = x1 + x2
        elif self.fusion_f == "ew":
            x_fused = x1 * x2
        elif self.fusion_f == "att":
            x_fused = self.attention_fusion(x1, x2)
        elif self.fusion_f == 'mlp':
            x_fused = self.mlp_fusion(x1, x2)
        else:
            raise ValueError("Unsupported fusion function")
        
        # Final reduction and classification
        pooled_x = self.pool(x_fused)
        pooled_x = self.dropout_final(pooled_x)  # Apply dropout before final linear layer
        pooled_x1 =  self.dropout_final(self.pool(x1))
        pooled_x2 = self.dropout_final(self.pool(x2))
        ## 
        if self.is_training:
            return self.final_linear(self.flatten(pooled_x)),self.unimodal_linear1(self.flatten(pooled_x1)),self.unimodal_linear2(self.flatten(pooled_x2))
        else:
            return self.final_linear(self.flatten(pooled_x))

    
def apply_umap(tensor, target_dim=2):
    """
    使用 UMAP 对输入张量进行降维
    :param tensor: 输入的 PyTorch 张量，形状为 (batch, channel, h, w)
    :param target_dim: 降维后的目标维度，默认为 2
    :return: 降维后的 NumPy 数组，形状为 (batch, target_dim)
    """
    # 确保 tensor 是 4D 张量 (batch, channel, h, w)
    if tensor.ndim != 4:
        raise ValueError(f"Input tensor must be 4-dimensional, but got {tensor.ndim} dimensions")

    # 获取批次大小、通道数、高度和宽度
    batch_size, channel, h, w = tensor.shape

    # 将 tensor 展平为 (batch, features)
    tensor_flat = tensor.view(batch_size, -1).cpu().detach().numpy()  # (batch, channel * h * w)

    # 使用 UMAP 进行降维
    umap_reducer = umap.UMAP(n_components=target_dim)
    tensor_reduced = umap_reducer.fit_transform(tensor_flat)

    return tensor_reduced
    
class VisualInteractSpikeNet_L(nn.Module):
    def __init__(self, pool_kernel_size=16, input_width=64, in_channels=[3, 1], num_classes=10, connect_f='ADD', fusion_f="concat"):
        super(VisualInteractSpikeNet_L, self).__init__()
        self.fusion_f = fusion_f
        
        # Input channel initialization
        in1, in2 = in_channels
        out_channels = 64
        
        # Stage 1
        self.init_conv1_1 = conv3x3(in1, out_channels)
        self.init_conv1_2 = conv3x3(in2, out_channels)
        self.interact_cell1_1 = InteractCell(out_channels, out_channels, connect_f)
        self.interact_cell1_2 = InteractCell(out_channels, out_channels, connect_f)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Stage 2
        self.conv2_1 = conv3x3(out_channels, 128)
        self.conv2_2 = conv3x3(out_channels, 128)
        self.interact_cell2_1 = InteractCell(128, 128, connect_f)
        self.interact_cell2_2 = InteractCell(128, 128, connect_f)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.dropout2 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Stage 3
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(128, 256)
        self.interact_cell3_1 = InteractCell(256, 256, connect_f)
        self.interact_cell3_2 = InteractCell(256, 256, connect_f)
        self.pool3 = nn.AvgPool2d(2, stride=2)
        self.dropout3 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Stage 4
        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(256, 512)
        self.interact_cell4_1 = InteractCell(512, 512, connect_f)
        self.pool4 = nn.AvgPool2d(2, stride=2)
        self.dropout4 = nn.Dropout(p=0.3)  # Add dropout after pooling
        
        # Final pooling and classification layer
        self.pool = nn.AvgPool2d(4, stride=4)
        
        # Adjust final_channels based on fusion method
        if self.fusion_f == "concat":
            final_channels = 512 * 2
        else:
            final_channels = 512
        
        if self.fusion_f == "att":
            self.attention_fusion = AttentionFusion(512)
        if self.fusion_f == 'mlp':
            self.mlp_fusion = MLPFusion(512, 512)
        
        final_width = input_width // (2 * 2 * 2 * 2 * 4)
        self.flatten = nn.Flatten()
        self.dropout_final = nn.Dropout(p=0.3)  # Add dropout before the final linear layer
        self.final_linear = nn.Linear(final_channels * final_width * final_width, num_classes)

    def forward(self, x1, x2):
        
        ## Stage 1
        x1 = self.init_conv1_1(x1)
        x2 = self.init_conv1_2(x2)
        x1, x2 = self.interact_cell1_1(x1, x2)   
        x1, x2 = self.interact_cell1_2(x1, x2)        
        x1, x2 = self.pool1(x1), self.pool1(x2)
        x1, x2 = self.dropout1(x1), self.dropout1(x2)  # Apply dropout after pool1

        ## Stage 2 
        x1, x2 = self.conv2_1(x1), self.conv2_2(x2)
        x1, x2 = self.interact_cell2_1(x1, x2)
        x1, x2 = self.interact_cell2_2(x1, x2)
        x1, x2 = self.pool2(x1), self.pool2(x2)
        x1, x2 = self.dropout2(x1), self.dropout2(x2)  # Apply dropout after pool2
        
        ## Stage 3
        x1, x2 = self.conv3_1(x1), self.conv3_2(x2)
        x1, x2 = self.interact_cell3_1(x1, x2)
        x1, x2 = self.interact_cell3_2(x1, x2)
        x1, x2 = self.pool3(x1), self.pool3(x2)
        x1, x2 = self.dropout3(x1), self.dropout3(x2)  # Apply dropout after pool3
        
        ## Stage 4
        x1, x2 = self.conv4_1(x1), self.conv4_2(x2)
        x1, x2 = self.interact_cell4_1(x1, x2)
        x1, x2 = self.pool4(x1), self.pool4(x2)
        x1, x2 = self.dropout4(x1), self.dropout4(x2)  # Apply dropout after pool4
        reduce_x1_2 = x1.cpu().detach().numpy() 
        reduce_x2_2 = x2.cpu().detach().numpy() 
        ## x1 (batch,channel,h,w),x2 (batch,channel,h,w)
        # reduce_x1_2 = apply_umap(x1,target_dim=2)
        # reduce_x2_2 = apply_umap(x2,target_dim=2)
        # Apply the fusion function
        if self.fusion_f == "concat":
            x_fused = torch.cat((x1, x2), dim=1)
        elif self.fusion_f == "sum":
            x_fused = x1 + x2
        elif self.fusion_f == "ew":
            x_fused = x1 * x2
        elif self.fusion_f == "att":
            x_fused = self.attention_fusion(x1, x2)
        elif self.fusion_f == 'mlp':
            x_fused = self.mlp_fusion(x1, x2)
        else:
            raise ValueError("Unsupported fusion function")
        
        # Final reduction and classification
        pooled_x = self.pool(x_fused)
        pooled_x = self.dropout_final(pooled_x)  # Apply dropout before final linear layer
        
        reduce_combine_2 = pooled_x.cpu().detach().numpy()
        
        return self.final_linear(self.flatten(pooled_x)),reduce_x1_2,reduce_x2_2,reduce_combine_2