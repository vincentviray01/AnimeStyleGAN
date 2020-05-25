import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as Datasets
from PIL import Image
import glob
from pathlib import Path


# class CustomDataset(Dataset):
#     def __init__(self, path_name):
#         super().__init__()
#         self.images = []
#         for name in glob.glob('data/images/jpg/*'):
#             self.images.append(name)
#     def __len__(self):
#         return len(self.images);
#     def __getitem__(self, index):
#         Image.open(self.images[index]).show()


def getDataLoader(args):
    """
    Loads the data loader for StyleGAN2 and applies preprocessing steps to it
    :param pathname: the name of the path to the folder containing the data
    :param args: command line arguments
    :return: the custom dataset
    """
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    customDataset = Datasets.ImageFolder(root = 'data/images', transform = transform)
    dataLoader = DataLoader(customDataset, batch_size = args.batch_size, shuffle = True)

    return dataLoader