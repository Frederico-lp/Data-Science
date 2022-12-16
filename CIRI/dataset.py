from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.image as mpimg
from PIL import Image


class IncidentsDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.target_transform = target_transform
        self.transform = transform
        self.labels = np.array(next(os.walk(img_dir))[1])
        self.data, self.targets = self.__load_data(img_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __load_data(self, directory):
        print("test")
        images, targets = [], []

        for target in np.arange(len(self.labels)):
            path = directory + "/" + self.labels[target]
            file_names = os.listdir(path=path)

            for filepath in file_names:
                if (filepath.endswith(".jpg") or filepath.endswith(".jpeg")) and not filepath.startswith("."):
                    img = Image.open(path + "/" + filepath).convert('RGB')
                    # img = mpimg.imread(path + "/" + filepath)
                    if self.transform:
                        img = self.transform(img)
                    images.append(img)
                    targets.append(target)

        return np.array(images), np.array(targets)

