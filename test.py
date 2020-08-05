import torch
import utils
import models
from torch.autograd import grad
class Mask(torch.nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        # self.weights = torch.nn.Parameter(torch.randn((2, 3, 4, 4)))
        self.x = torch.nn.Conv2d(3, 5, 3)
    def forward(self, noise):
        return torch.nn.Linear(2, 5)(noise)

import os
# print(sorted(os.listdir('saves'), key = lambda x: int(x[4: -3])))


import gzip
f = gzip.open('data/mnist/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 60000

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

import matplotlib.pyplot as plt
images = np.asarray(data)

print(images[5999])

plt.savefig('foo.png')

plt.imshow(images[0].squeeze())
plt.show()
import torchvision
torchvision.datasets.MNIST("data/mnist", train=True, transform=None, target_transform=None, download=True)
# x = torch.randn(4, 2, 2, 3)
# k = [1 , 2, 3]
#
# print(x)
# for a, b in zip(x, k):
#     print(a)
#     print(b)

# print(utils.createStyleMixedNoiseList(1, 4, 5, models.MappingNetwork(), 'cpu').size())
# print(utils.createStyleMixedNoiseList(1, 4, 5, models.MappingNetwork(), 'cpu')[:, 0:3, 1:3])

# print(torch.sqrt(torch.sum(torch.square(torch.randn(3,5, 5)))))
# if 1 == 3:
#     x = 2
# print(x)
# lossfn = torch.nn.MSELoss()
# x = Mask()
# noise = torch.ones([1, 3, 4, 2], requires_grad=True)
# print(noise.requires_grad)
# truths = torch.zeros([1, 3, 4, 5])
# noise = x(noise)
# # print(noise)
# # print(noise2)
# # print(noise2.size())
# loss = lossfn(noise, truths )
# print("===================================================")
# # print(noise.grad)
# # print(grad(outputs=loss, inputs = noise)[0].size())
# p = grad(outputs=loss, inputs = noise, retain_graph=True)[0].view(2, -1)
# grad_penalty = (p.norm(2, dim = 1)**2).mean()
# print(grad_penalty)
# grad_penalty2 = torch.norm(torch.sum(p.square(), axis = 1), p=2)
# grad_penalty3 = torch.sum(p.square(), axis = 1).mean()
# grad_penalty4 = torch.sum(p.square(), axis = 1)**0.5
# print(grad_penalty2)
# print(grad_penalty3)
# print(grad_penalty4)
# print(p)
