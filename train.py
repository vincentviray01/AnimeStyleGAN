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
parser.add_argument ('-e', '--epochs', type=int, required=True, help="The number of epochs to train for")
group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verbose', action='store_true', help ='print verbose')
args = parser.parse_args()

image_size = args.img_size
batch_size = args.batch_size
epochs = args.epochs
quiet = args.quiet == True
verbose = not quiet
latent_dim = 256
mixed_probability = 0.9
discriminator_filters = 8
generator_filters = 8
pl_beta = 0.99

if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)
    #model = StyleGan2Model()
    dataLoader = utils.getDataLoader(batch_size, image_size)
    # print(len(dataLoader) / 10)
    Trainer = models.Trainer(batch_size, image_size, latent_dim, epochs, discriminator_filters, generator_filters, device, mixed_probability, pl_beta)
    print("Apex available: ", Trainer.apex_available)
    # Trainer.resetSaves()
    # x, y = next(enumerate(dataLoader))
    # x, y = next(enumerate(dataLoader))
    # print(y[0].size())
    # for x in range(10):
    #     try:
    #         Trainer.train()
    #     except Exception as e:
    #         print(e)
    #         torch.cuda.empty_cache()
    #     print("one iteration")
    Trainer.train()
    print("DONE TRAINING")