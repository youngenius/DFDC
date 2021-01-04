#METHOD 1
import torch
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from PIL import Image
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path):
        self.folder_name = os.listdir(root_path)
        self.root = root_path
        self.json_paths = os.path.join(root_path, 'metadata.json') # 1
        with open(self.json_paths) as json_file:
            self.json_data = json.load(json_file)

        self.transform = transforms.Compose([
                transforms.Scale(256),
                transforms.ToTensor(),
        ])

        self.len = len(self.folder_name) #folder len

    def __getitem__(self, index):
        self.image_paths = Path(os.path.join(self.root, self.folder_name[index])).rglob('*.jpg') # 1folder 9images
        image_paths = list(self.image_paths)
        index_jump = len(image_paths)//15
        image_9 = []
        for index_image in range(0, len(image_paths), index_jump):
            x = Image.open(image_paths[index_image])
            image_9.append(self.transform(x))
        #json
        self.label = self.json_data[self.folder_name[index]+'.mp4']['label']

        return torch.cat(image_9), self.label

    def __len__(self):
        return self.len

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_path):
        self.folder_name = os.listdir(root_path)
        self.root = root_path
        self.json_paths = os.path.join(root_path, 'metadata.json')  # 1
        with open(self.json_paths) as json_file:
            self.json_data = json.load(json_file)

        self.transform = transforms.Compose([
            transforms.Scale(256),
            transforms.ToTensor(),
        ])

        self.len = len(self.folder_name)  # folder len

    def __getitem__(self, index):
        self.image_paths = Path(os.path.join(self.root, self.folder_name[index])).rglob('*.jpg')  # 1folder 9images
        image_paths = list(self.image_paths)
        index_jump = len(image_paths) // 15
        image_9 = []
        for index_image in range(0, len(image_paths), index_jump):
            x = Image.open(image_paths[index_image])
            image_9.append(self.transform(x))
        # json
        self.label = self.json_data[self.folder_name[index] + '.mp4']['label']

        return torch.cat(image_9), self.label

    def __len__(self):
        return self.len