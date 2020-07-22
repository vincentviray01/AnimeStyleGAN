import utils
import argparse
import torch
import models
from PIL import Image
from matplotlib import pyplot as plt
import tqdm

parser = argparse.ArgumentParser(description="Train StyleGAN2")
parser.add_argument('-i', '--img_size', type=int, metavar='', required=True, help="The resolution of the training and generated images")
parser.add_argument('-b', '--batch_size', type=int, metavar='', required=True, help="Batch size when training")
parser.add_argument ('-e' '--epochs', type=int, required=True, help="The number of epochs to train for")
group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verbose', action='store_true', help ='print verbose')
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #model = StyleGan2Model()
    dataLoader = utils.getDataLoader(args)
    x, y = next(enumerate(dataLoader))
    # utils.showImage(y[0].view(3,2,1,0)
    print(x)
    utils.showImage(y[0])
    print(y[0].size())
    # print(args)
    for epoch in range(2):
        pass
    # d = models.MappingNetwork(512, 512, 3)
    #
    # noise = utils.createNoise(1, 512)
    # print(noise)
    # # print(d)
    # noised = d(noise)
    # print(noised.size())