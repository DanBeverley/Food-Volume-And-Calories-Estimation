"""Defines a data pipeline for training and testing a depth estimation model in PyTorch,
where each sample consists of an image and its corresponding depth map"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyo_transform import *

class depthDataset(Dataset):
    """Face Landmarks dataset
    Loads the CSV file containing file paths for the images and depth maps.
    Saves any transformations passed as an argument to apply to the data later."""
    def __init__(self, csv_file, transform = None):
        self.frame = pd.read_csv(csv_file, header = None)
        self.transform = transform
    def __getitem__(self, idx):
        """Retrieves an image and its corresponding depth map by index."""
        image_name = self.frame.iloc[idx, 0]
        depth_name = self.frame.iloc[idx, 1]

        image = Image.open(image_name)
        depth = Image.open(depth_name)

        sample = {"image":image, "depth": depth}

        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        """Returns the length of the dataset, which is the number of entries"""
        return len(self.frame)

def getTrainingData(batch_size = 64):
    """Creates a DataLoader for the training dataset.
    Applies a series of transformations for data augmentation, which helps the model generalize better."""
    __imagenet_pca = {"eigval":torch.Tensor([.2715, .0188, .0045]),
                      "eigvec":torch.Tensor([
                          [-.5675, .7192, .4009],
                          [-.5808, .0045, .8140],
                          [-.5836, -.6948, .4203]
                      ])}
    __imagenet_stats = {"mean":[.485, .456, .406],
                        "std":[.229, .224, .225]}
    transformed_training = depthDataset(csv_file="./data/nyu2_train.csv",
                                        transform = transform.Compose([
                                            Scale(240), RandomHorizontalFlip(),
                                            RandomRotate(5), CenterCrop([304, 228],[152,114]),
                                            ToTensor(), Lighting(.1, __imagenet_pca["eigval"],
                                                                 __imagenet_pca["eigvec"]),
                                            ColorJitter(brightness=.4,
                                                        contrast=.4,
                                                        saturation=.4),
                                            Normalize(__imagenet_stats["mean"],
                                                      __imagenet_stats["std"])
                                        ]))
    dataloader_training = DataLoader(transformed_training, batch_size, shuffle = True,
                                     num_workers = os.cpu_count(), pin_memory = True)
    return dataloader_training

def getTestingData(batch_size=64):
    """Prepares a DataLoader for the test dataset with similar preprocessing transformations
       but excludes augmentations that could alter the test data."""
    __imagenet_stats = {"mean":[.485, .456, .406],
                        "std":[.229, .224, .225]}
    # scale = random.uniform(1,1.5)
    transformed_testing = depthDataset(csv_file = "./data/nyu2_test.csv",
                                       transform = transforms.Compose([Scale(240),
                                                                       CenterCrop([304,228],[304, 228]),
                                                                       ToTensor(is_test = True),
                                                                       Normalize(__imagenet_stats["mean"],
                                                                                 __imagenet_stats["std"])]))
    dataloader_training = DataLoader(transformed_testing, batch_size, shuffle = False,
                                     num_workers = 0, pin_memory = False)
    return dataloader_training


