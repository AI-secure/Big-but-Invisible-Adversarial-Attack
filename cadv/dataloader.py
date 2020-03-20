import numpy as np
import os
import random
import itertools
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class im_dataset(Dataset):
    def __init__(self, data_dir, im_size=224):
        self.data_dir = data_dir
        self.imgpaths = self.get_imgpaths()

        self.transform = transforms.Compose([
                       transforms.Resize((im_size, im_size)),
                       transforms.ToTensor()])

    def get_imgpaths(self):
        paths = [os.path.join(self.data_dir, x) for x in os.listdir(self.data_dir)  if x.endswith(('JPEG','jpg', 'png'))]
        return paths
    
    def __getitem__(self, idx):
        img_name = self.imgpaths[idx]
        file_name = os.path.splitext(os.path.basename(img_name))[0]
        image = Image.open(img_name)
        image_t = self.transform(image)
        return image_t, file_name

    def __len__(self):
        return len(self.imgpaths)

