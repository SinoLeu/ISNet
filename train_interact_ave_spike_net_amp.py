from torch.utils.data import Dataset, DataLoader
from modules.neuron import MultiLIFNeuron
import os
from PIL import Image
import torchaudio
from torchvision import transforms
from torchaudio import transforms as aT
from torch.utils.data import random_split, DataLoader
import torch
from dataset_loader import VisualAudioDataSet,CREMADVisualAudioDataSet,AVEVisualAudioDataSet
import time
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, surrogate as surrogate_sj
import argparse
from tqdm import tqdm
from models.spiking_interact import InteractSpikeNet_S,InteractSpikeNet_L,InteractSpikeNet_XL
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.optim.lr_scheduler import StepLR


def save_model_if_best(current_acc, best_acc, model, save_path):
    if current_acc > best_acc:
        save_path = save_path + '_save_best_model.pth'
        best_acc = current_acc
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved with accuracy: {best_acc:.4f}")
    return best_acc


## python train_interact_cremad_spike_net_amp.py --model_scale L --train_data AVE_dataset --dataset_path ./AVE_dataset --t_step 5 --epoch 50 --n_mels 96  --batch_size 16 --pool_kern_size 16 --lr 1e-4 --fusion_method concat > interact_ave.log 2>&1


parser = argparse.ArgumentParser(description='Train a spiking neural network model on audio-visual data.')
parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size')
parser.add_argument('--n_mels', type=int, help='Number of mel bands')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
parser.add_argument('--epoch', type=int, default=15, help='Number of training epochs')
parser.add_argument('--t_step', type=int, default=5, help='Number of time steps for training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu)')
parser.add_argument('--train_data', type=str, default='CIFAR10-AV', help='training dataset')
parser.add_argument('--dataset_path', type=str, default='CIFAR10-AV', help='training dataset')
parser.add_argument('--split_size', type=float,default=0.7, help='train test split size')
parser.add_argument('--pool_kern_size', type=int, default=4, help='Number of time steps for training')
parser.add_argument('--model_scale', type=str, default='S', help='the model scale')
## concat,sum,ew,att
parser.add_argument('--fusion_method',type=str, default='concat', help='the model scale')
parser.add_argument('--load_trained_model',type=str, default='', help='the model scale')
args = parser.parse_args()


# Use the device specified in command line arguments
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


audio_transforms = torch.nn.Sequential(
    aT.MelSpectrogram(n_fft=args.n_fft, hop_length=None, n_mels=args.n_mels),
    aT.AmplitudeToDB(),
    transforms.Resize((args.n_mels, args.n_mels))
)

transform = transforms.Compose([
    transforms.Resize((args.n_mels, args.n_mels)),
    transforms.ToTensor()
])

train_dataset = AVEVisualAudioDataSet(args.dataset_path, transform=transform, audio_transform=audio_transforms,mode='train')
test_dataset = AVEVisualAudioDataSet(args.dataset_path, transform=transform, audio_transform=audio_transforms,mode='test')


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

if args.model_scale == 'S':
    net = InteractSpikeNet_S(pool_kernel_size=args.pool_kern_size,input_width=args.n_mels,in_channels=[3,2],num_classes=28,connect_f='ADD',fusion_f=args.fusion_method)
elif args.model_scale == 'L':
    net = InteractSpikeNet_L(pool_kernel_size=args.pool_kern_size,input_width=args.n_mels,in_channels=[3,2],num_classes=28,connect_f='ADD',fusion_f=args.fusion_method)
elif args.model_scale == 'XL':
    net = InteractSpikeNet_XL(pool_kernel_size=args.pool_kern_size,input_width=args.n_mels,in_channels=[3,2],num_classes=28,connect_f='ADD',fusion_f=args.fusion_method)

def count_model_parameters(model):
    # 获取所有参数，但排除self.unimodal_linear1和self.unimodal_linear2的参数
    total_params = sum(p.numel() for name, p in model.named_parameters() 
                       if 'unimodal_linear1' not in name and 'unimodal_linear2' not in name)
    total_params_million = total_params / 1e6
    return total_params_million



total_params_million = count_model_parameters(net)
print(f'Total number of parameters: {total_params_million:.2f}M')

checkpoint_path = args.load_trained_model
if os.path.exists(checkpoint_path):
    net.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    print(f"Checkpoint loaded successfully from '{checkpoint_path}'")
else:
    print(f"Checkpoint '{checkpoint_path}' not found.")
    
scaler = GradScaler()
net.to(args.device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []


best_acc = 0.5572  # 初始最佳准确率

for epoch_index in range(args.epoch):
    net.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch_index + 1}/{args.epoch}', unit='batch') as pbar:
        for img1, img2, label in train_loader:
            optimizer.zero_grad()
            img1, img2, label = img1.to(args.device), img2.to(args.device), label.to(args.device)
            label_onehot = F.one_hot(label, 28).float()
            with autocast():
                out_fr = 0.
                out_fr1 = 0.
                out_fr2 = 0.
                for t in range(args.t_step):
                    out_fr,out_fr1,out_fr2 = net(img1[:,t], img2)
                    out_fr += out_fr
                    out_fr1 += out_fr1
                    out_fr2 += out_fr2
                out_fr = out_fr / args.t_step
                out_fr1 = out_fr1 / args.t_step
                out_fr2 = out_fr2 / args.t_step
                loss = F.mse_loss(out_fr, label_onehot)
                loss1 = F.mse_loss(out_fr1, label_onehot)
                loss2 = F.mse_loss(out_fr2, label_onehot)
                loss = loss + 0.5*loss1 + 0.5*loss2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()
            batch_acc = (out_fr.argmax(1) == label).float().mean().item()
            functional.reset_net(net)
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.4f}'})
        
        train_loss /= train_samples
        train_acc /= train_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
    
    test_loss = 0
    test_acc = 0
    test_samples = 0
    t_step = args.t_step
    
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Validation', unit='batch', leave=False) as pbar:
            for img1, img2, label in test_loader:
                
                img1 = img1.to(device)
                img2 = img2.to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, 28).float()
                
                with autocast():
                    out_fr = 0.
                    for t in range(args.t_step):
                        out_fr,_,_ = net(img1[:,t], img2)
                        out_fr += out_fr
                    out_fr = out_fr / args.t_step
                    loss = F.mse_loss(out_fr, label_onehot)
                
                out_fr = out_fr / args.t_step
                loss = F.mse_loss(out_fr, label_onehot)
                
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                functional.reset_net(net)
        
        save_path = f'{args.train_data}_{args.model_scale}_{args.fusion_method}'
        test_loss /= test_samples
        test_acc /= test_samples
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')
        best_acc = save_model_if_best(test_acc, best_acc, net, save_path)
    # 每个epoch结束时更新学习率
    if epoch_index >= 39:  # 第40个epoch开始衰减
        scheduler.step()