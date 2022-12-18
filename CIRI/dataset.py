from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class IncidentsDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.target_transform = target_transform
        self.transform = transform
        self.labels = np.array(next(os.walk(img_dir))[1])
        # self.data, self.targets = self.__load_data(img_dir)
        self.image_paths, self.targets = self.__get_paths_and_targets(img_dir)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, self.targets[index]

    def get_item_with_target(self, target, index):
        label = self.labels[target]

        path = self.image_paths[[label in x for x in self.image_paths]][index]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img

    def __get_paths_and_targets(self, img_dir):
        paths, targets = [], []

        for t in np.arange(len(self.labels)):
            path = img_dir + "/" + self.labels[t]
            file_names = os.listdir(path)
            for name in file_names:
                if (name.endswith(".jpg") or name.endswith(".jpeg")) and not name.startswith("."):
                    paths.append(path + "/" + name)
                    targets.append(t)

        return np.array(paths), np.array(targets)
