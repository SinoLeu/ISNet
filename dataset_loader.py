from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchaudio
from torchvision import transforms
from torchaudio import transforms as aT
from torch.utils.data import random_split, DataLoader
import torch
from PIL import Image

def adjust_waveform_length(waveform, target_length):
    current_length = waveform.size(-1)
    if current_length < target_length:
        # 需要填充
        padding_size = target_length - current_length
        pad = torch.zeros(waveform.size(0), padding_size)  # 填充零
        waveform = torch.cat((waveform, pad), dim=1)  # 在最后一个维度上连接
    elif current_length > target_length:
        # 需要截断
        waveform = waveform[:, :target_length]  # 截断超出部分
    return waveform

class CREMADVisualAudioDataSet(Dataset):
    def __init__(self, root_dir,  mode='train',transform=None, audio_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.audio_transform = audio_transform
        self.visual_path = "ImageFrames"
        self.audio_path = "AudioWAV"
        if mode == "train":
            self.read_txt = "train_cre.txt"
        else:
            self.read_txt = "test_cre.txt"
        self.stat_cre = [
            "Anger","Disgust","Fear","Happy","Neutral","Sad" ]
        self.stat_cre_mapping = {label:i for i,label in enumerate(self.stat_cre)}
        self.load_names_label = []
        with open(os.path.join(self.root_dir,self.read_txt), 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split()
            name = parts[0].split(".")[0]
            label = parts[1]
            self.load_names_label.append((name,self.stat_cre_mapping[label]))
    
    def __len__(self):
        return len(self.load_names_label)
    
    def __getitem__(self, idx):
        name, label = self.load_names_label[idx]
        video_names = f"{name}"
        audio_name = f"{name}.wav"
        video_read_path = os.path.join(self.root_dir,os.path.join(self.visual_path,video_names))
        audio_read_path = os.path.join(self.root_dir,os.path.join(self.audio_path,audio_name))
        image_arr = []
        for frame_name in os.listdir(video_read_path):
            frame_path = os.path.join(video_read_path,frame_name)
            image = Image.open(frame_path).convert('RGB')
            image = self.transform(image)
            image = image.float()
            image_arr.append(image)
        ## 
        waveform, sample_rate = torchaudio.load(audio_read_path)
        ## 
        if self.audio_transform:
            waveform = self.audio_transform(waveform)
        ##
        # image_n = torch.cat(image_arr, dim=1)
        image_n = torch.stack(image_arr, dim=0)
        return image_n, waveform, label
        
## 
class VisualAudioDataSet(Dataset):
    def __init__(self, root_dir, transform=None, audio_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.audio_transform = audio_transform
        self.data = []
        self.classes = []
        visual_dir = os.path.join(root_dir, 'vision')
        audio_dir = os.path.join(root_dir, 'audio')

        for category in sorted(os.listdir(visual_dir)):
            visual_category_path = os.path.join(visual_dir, category)
            audio_category_path = os.path.join(audio_dir, category)
            if os.path.isdir(visual_category_path) and os.path.isdir(audio_category_path):
                for visual_file in os.listdir(visual_category_path):
                    # if visual_file.endswith('.png'):
                    if visual_file.endswith('.jpg'):
                        base_filename = os.path.splitext(visual_file)[0]
                        audio_file = f"{base_filename}.wav"
                        visual_path = os.path.join(visual_category_path, visual_file)
                        audio_path = os.path.join(audio_category_path, audio_file)
                        if os.path.exists(audio_path):
                            self.data.append((visual_path, audio_path, category))

        self.classes = sorted(set([item[2] for item in self.data]))
        
    ##
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        visual_path, audio_path, category = self.data[idx]

        image = Image.open(visual_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.audio_transform:
            waveform = self.audio_transform(waveform)

        label = self.classes.index(category)
        return image, waveform, label

class AVEVisualAudioDataSet(Dataset):
    def __init__(self, root_dir,  mode='train',transform=None, audio_transform=None,target_length=47000):
        self.root_dir = root_dir
        self.transform = transform
        self.audio_transform = audio_transform
        self.visual_path = "AVE_vision"
        self.audio_path = "AVE_audio"
        self.target_length = target_length
        if mode == "train":
            self.read_txt = "train_ave.txt"
        elif mode == 'test':
            self.read_txt = "test_ave.txt"
        else:
            self.read_txt = "val_ave.txt"

        self.stat_cre = ['Church bell', 'Male speech, man speaking', 'Bark',
       'Fixed-wing aircraft, airplane', 'Race car, auto racing',
       'Female speech, woman speaking', 'Helicopter', 'Violin, fiddle',
       'Flute', 'Ukulele', 'Frying (food)', 'Truck', 'Shofar',
       'Motorcycle', 'Acoustic guitar', 'Train horn', 'Clock', 'Banjo',
       'Goat', 'Baby cry, infant cry', 'Bus', 'Chainsaw', 'Cat', 'Horse',
       'Toilet flush', 'Rodents, rats, mice', 'Accordion', 'Mandolin']
        self.stat_cre_mapping = {label:i for i,label in enumerate(self.stat_cre)}
        self.load_names_label = []
        with open(os.path.join(self.root_dir,self.read_txt), 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split("&")
            name = parts[0]
            label = parts[1]
            self.load_names_label.append((name,self.stat_cre_mapping[label]))
    
    def __len__(self):
        return len(self.load_names_label)
    
    def __getitem__(self, idx):
        name, label = self.load_names_label[idx]
        video_names = f"{name}"
        audio_name = f"{name}.wav"
        video_read_path = os.path.join(self.root_dir,os.path.join(self.visual_path,video_names))
        audio_read_path = os.path.join(self.root_dir,os.path.join(self.audio_path,audio_name))
        image_arr = []
        for frame_name in os.listdir(video_read_path):
            frame_path = os.path.join(video_read_path,frame_name)
            image = Image.open(frame_path).convert('RGB')
            image = self.transform(image)
            image = image.float()
            image_arr.append(image)
        waveform, sample_rate = torchaudio.load(audio_read_path)
        waveform = adjust_waveform_length(waveform, self.target_length)
        if self.audio_transform:
            waveform = self.audio_transform(waveform)
        # waveform = waveform.mean(dim=0, keepdim=True)
        image_n = torch.stack(image_arr, dim=0)
        return image_n, waveform, label

## load CIFAR10-AV

# # 使用示例
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor()
# ])


## load UrbanSound8K-AV
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# n_fft = 2048  # FFT 的窗口大小
# hop_length = None# 帧移
# n_mels = 64  # 梅尔滤波器的数量
# # 创建一个音频处理流水线
# audio_transforms = torch.nn.Sequential(
#     aT.MelSpectrogram( n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
#     aT.AmplitudeToDB()
# )


# dataset = VisualAudioDataSet('./', transform=transform, audio_transform=audio_transforms)
# # data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

