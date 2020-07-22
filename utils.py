import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as Datasets
from matplotlib import pyplot as plt
import numpy as np
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

def showImage(image):
    """
    Shows image to screen by removing batch dimention and permuting the pytorch tensor to fit pyplot standards
    :param image: image to show to the screen
    :return: void
    """
    plt.axis("off")
    plt.imshow(image[0].squeeze(0).permute(1, 2, 0))
    plt.show()

def createNoise(batch_size, latent_size):
    return torch.empty(batch_size, latent_size).normal_(mean=0, std=0.5)#np.random.normal(0, 1, size = [batch_size, latent_size]).astype('float32')

def createNoiseList(batch_size, latent_size, num_layers):
    return [createNoise(batch_size, latent_size)] * num_layers

def createMixedNoise(batch_size, latent_size, num_layers):
    randomCut = np.random.randint(num_layers)
    first_part = [createNoise(batch_size, latent_size)]* randomCut
    second_part = [createNoise(batch_size, latent_size)] * (num_layers - randomCut)
    return first_part + second_part


def gradientPenalty(predictions, samples):  #TODO
    predictions.backward(samples)
    gradients = predictions.grad
    gradients_square = gradients**2
    gradients_penalty = np.sum(gradients_square, axis=0)

    return torch.norm(gradients_penalty)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)




