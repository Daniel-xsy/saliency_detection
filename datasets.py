import os
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root, img_size=256, split='train', transform=None):
        super(ImageDataset, self).__init__()

        self.root = root
        self.split = split
        self.img_size =img_size
        self.transform = transform
        self.maps_transform = transforms.Compose([
        transforms.Resize((256,256), interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ])
        self.image_files = os.listdir(os.path.join(root, 'images', split))
        self.maps_files = os.listdir(os.path.join(root, 'maps', split))
        list.sort(self.image_files)
        list.sort(self.maps_files)
        assert len(self.image_files) == len(self.maps_files)

    def _getitem(self, id):
        image_file = self.image_files[id]
        map_file = self.maps_files[id]
        assert image_file.split('.')[0] == map_file.split('.')[0]

        img = Image.open(os.path.join(self.root, 'images', self.split, image_file))
        img = img.convert('RGB')
        map = Image.open(os.path.join(self.root, 'maps', self.split, map_file))

        if self.transform is not None:
            img = self.transform(img)
            if self.maps_transform is not None:
                map = self.maps_transform(map)

        return img, map

    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return len(self.image_files)