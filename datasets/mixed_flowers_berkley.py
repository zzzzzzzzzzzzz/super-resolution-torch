# coding: utf-8
import os
from torch.utils.data import Dataset
from PIL import Image


class MixedFlowersBerkley(Dataset):
    """
    Dataset, which was made by mixing flowers and berkley sets
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir, self.dirs, self.files = os.walk(root_dir).__next__()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.files[idx]))
        sample = {'lr': img, 'hr': img}

        if self.transform:
            sample = self.transform(sample)

        return sample['lr'], sample['hr']
