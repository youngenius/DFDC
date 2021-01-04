#METHOD 2
import torch
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from PIL import Image
from pathlib import Path
import pandas as pd
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Resize
from albumentations.pytorch.functional import img_to_tensor
from torchvision.utils import save_image
import cv2
from albu import IsotropicResize
size = 256
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path):
        self.folder_name = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]#os.listdir(root_path)[:-1]
        self.root = root_path
        #self.image_paths = list(Path(self.root).rglob('*.jpg'))
        self.json_paths = os.path.join(root_path, 'metadata.json') # 1
        with open(self.json_paths) as json_file:
            self.json_data = json.load(json_file)
        self.transform = Compose([
            Resize(size, size),
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(p=0.5),
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=0.7),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
            ToGray(p=0.1),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ])
        self.normalize = {"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]}
        #self.len = len(self.image_paths) #folder len
        self.len = len(self.folder_name)

    def __getitem__(self, index):
        self.image_paths = Path(os.path.join(self.root, self.folder_name[index])).rglob('*.jpg')
        image_paths = list(self.image_paths)
        index_jump = len(image_paths) / 32
        image = []
        index_image = 0
        while(index_image <len(image_paths)):
            x = cv2.imread(str(image_paths[int(index_image)]))#Image.open(image_paths[int(index_image)])
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=x)
            transformed_image = transformed["image"]
            image.append(img_to_tensor(transformed_image))
            index_image += index_jump
        '''
        for index_image in range(0, len(image_paths), index_jump):
            x = Image.open(image_paths[index_image])
            image.append(self.transform(x))
        '''
        #x = Image.open(self.image_paths[index])
        #label = self.json_data[str(self.image_paths[index]).split('/')[-2]+'.mp4']['label']
        label = self.json_data[self.folder_name[index] + '.mp4']['label']
        if label == 'FAKE':
            self.label = torch.tensor(1.0).repeat(32)
        else:
            self.label = torch.tensor(0.0).repeat(32)

        return torch.cat(image).view(-1, 3, 256, 256), self.label

    def __len__(self):
        return self.len

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_path):
        self.folder_name = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))] #os.listdir(root_path)[:-1]
        self.root = root_path
        #self.image_paths = list(Path(self.root).rglob('*.jpg'))
        self.json_path = os.path.join(root_path, 'metadata.json')  # 1
        with open(self.json_path) as json_file:
            self.json_data = json.load(json_file)
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
        ])
        #self.len = len(self.image_paths) #folder len
        self.len = len(self.folder_name)

    def __getitem__(self, index):
        self.image_paths = Path(os.path.join(self.root, self.folder_name[index])).rglob('*.jpg')
        image_paths = list(self.image_paths)
        index_jump = len(image_paths) / 32
        image = []
        index_image = 0
        while (index_image < len(image_paths)):
            x = Image.open(image_paths[int(index_image)])
            image.append(self.transform(x))
            index_image += index_jump
        #x = Image.open(self.image_paths[index])
        #self.label = self.json_data[str(self.image_paths[index]).split('/')[-2]+'.mp4']['is_fake']
        self.label = self.json_data[self.folder_name[index] + '.mp4']['is_fake']
        return torch.cat(image).view(-1, 3, 256, 256), self.label

    def __len__(self):
        return self.len