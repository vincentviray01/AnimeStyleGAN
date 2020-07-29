import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as Datasets
from matplotlib import pyplot as plt
import numpy as np
from torch.autograd import grad
from models import MappingNetwork
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


def getDataLoader(batch_size, image_size):
    """
    Loads the data loader for StyleGAN2 and applies preprocessing steps to it
    :param pathname: the name of the path to the folder containing the data
    :param args: command line arguments
    :return: the custom dataset
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    customDataset = Datasets.ImageFolder(root='data/images', transform=transform)
    dataLoader = DataLoader(customDataset, batch_size=batch_size, shuffle=True)

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


def create_image_noise(batch_size, image_size, device):
    return torch.FloatTensor(batch_size, image_size, image_size, 1).uniform_(0, 1).to(device)


def createNoise(batch_size, latent_size, device):
    return torch.empty(batch_size, latent_size).normal_(mean=0,
                                                        std=0.5).to(device)  # np.random.normal(0, 1, size = [batch_size, latent_size]).astype('float32')


def createStyleNoiseList(batch_size, latent_size, num_layers, StyleVectorizer, device):
    return StyleVectorizer(createNoise(batch_size, latent_size, device))[:, None, :].expand(-1, int(num_layers), -1)


def createStyleMixedNoiseList(batch_size, latent_size, num_layers, StyleVectorizer, device):
    randomCut = np.random.randint(num_layers)
    return torch.cat((createStyleNoiseList(batch_size, latent_size, randomCut, StyleVectorizer, device),
                      createStyleNoiseList(batch_size, latent_size, (num_layers - randomCut), StyleVectorizer, device)), dim=1)


def gradientPenalty(images, probability_of_real, device):
    gradients = grad(outputs=probability_of_real, inputs=images, create_graph = True, retain_graph=True).to(device)
    gradients = gradients.view(images.shape[0], -1)
    return torch.sum(gradients.square(), axis=1).mean()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool
