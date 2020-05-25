import utils
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description = "Train StyleGan2")
parser.add_argument('-i', '--img_size', type = int, metavar='', required = True, help = "The resolution of the training and generated images", )
parser.add_argument('-b', '--batch_size', type = int, metavar = '', required = True, help = "Batch size when training")
group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action = 'store_true', help='print quiet')
group.add_argument('-v', '--verbose', action = 'store_true', help ='print verbose')
args = parser.parse_args()

if __name__ == '__main__':
    #model = StyleGan2Model()
    dataLoader = utils.getDataLoader(args)
    print("hi")
    x, y = next(enumerate(dataLoader))