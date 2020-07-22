import torch

class Mask(torch.nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        # self.weights = torch.nn.Parameter(torch.randn((2, 3, 4, 4)))
        self.x = torch.nn.Conv2d(3, 5, 3)
x = Mask()
for j in x.parameters():
    # print(j)
    print()
    print(j.data)