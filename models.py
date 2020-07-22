import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import torch.optim as optim
import numpy as np

class Conv2DWeightDemod(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1,
                 dilation=1, padding = 0):
        super().__init__()
        l2_norm_weights = torch.sqrt(self.conv2d.weights)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.weight = nn.Parameter(torch.randn((output_channels, input_channels, kernel_size, kernel_size)))

    def forward(self, input_vector, style_vector):

        # Weight Demodulation
        weight = self.weight * style_vector
        demodulation_coefficient = torch.rsqrt(weight.pow(2).sum([1, 2, 3], keepdims = True) + 1e-8)
        weight *= demodulation_coefficient
        output = F.conv2d(input_vector, weight, stride = self.stride, passing = self.padding, dilation = self.dilation,
                          groups = 1)

        return nn.LeakyReLU(0.2)(output.reshape(-1, self.output_channels, self.height, self.weight))

class RGBBlock(nn.Module):
    def __init__(self, input_channels, rgb_upsample = True):
        # 3 into conv demod since RGB image has three channels
        super().__init__()
        self.rgb_upsample = rgb_upsample
        self.conv2dWeightDemod = Conv2DWeightDemod(input_channels, 3)

    def forward(self, image, prev_rgb, style_vector):
        image = self.conv2dWeightDemod(image, style_vector)
        if prev_rgb is not None:
            image = image + prev_rgb
        if self.rgb_upsample:
            image = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        return image

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 3, upsample = True, upsample_rgb = True, stride = 1, dilation = 1,
                 padding = 0):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.Conv2DWeightDemod = Conv2DWeightDemod(input_channels, output_channels)
        self.RGBBlock = RGBBlock()

    def forward(self, image, style_vector):
        if self.psample == True:
            image = nn.Upsample(scale_factor=2, mode='bilinear')(image)
        image = self.Conv2DWeightDemod(image + style_vector)

        return image, rgb



class DiscriminatorBlock(nn.Module):
    # will downsample using stride = 2
    def __init__(self, input_channels, output_channels, kernel_size = 3):
        super().__init__()
        # skip connection for res blocks
        self.skip = nn.Conv2d(input_channels, output_channels, 1, stride=2)
        self.mainLine = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, padding = 1),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(output_channels, output_channels, kernel_size, padding = 1, stride=2))


    def forward(self, input):
        skipped_line_input = self.skip(input)
        main_line_input = self.mainLine(input)
        return nn.LeakyReLU(0.2)((skipped_line_input + main_line_input) / math.sqrt(2))


class MappingNetwork(nn.Module):
    def __init__(self, in_channels = 512, latent_channels = 512, gain = 2 ** (0.5)):
        super().__init__()
        self.sequential = nn.Sequential(nn.Linear(in_channels, latent_channels), nn.LeakyReLU(0.2),
                                         nn.Linear(latent_channels, latent_channels), nn.LeakyReLU(0.2),
                                         nn.Linear(latent_channels, latent_channels), nn.LeakyReLU(0.2),
                                         nn.Linear(latent_channels, latent_channels), nn.LeakyReLU(0.2),
                                         nn.Linear(latent_channels, latent_channels), nn.LeakyReLU(0.2),
                                         nn.Linear(latent_channels, latent_channels), nn.LeakyReLU(0.2),
                                         nn.Linear(latent_channels, latent_channels), nn.LeakyReLU(0.2),
                                         nn.Linear(latent_channels, latent_channels), nn.LeakyReLU(0.2))
        self.sequential.apply(utils.init_weights)

    def forward(self, latent_z):
        return self.sequential(latent_z)

# The generator upsamples
class Generator(nn.Module):
    # gen filters/channels goes from channels noise size -> desired_features_generator * 16 ->
    # desired_features_generator * 8 -------> desired_feeatuers_generator
    def __init__(self, constant_vector, image_size, latent_dim,  starting_filter = 5):
        super().__init__()
        self.mappingNetwork = MappingNetwork(512, 512)
        self.style_vector = self.mappingNetwork(constant_vector)
        self.noise = utils.createNoise()
        self.start_of_network = False
        self.end_of_network = False
        self.generatorBlocks = []
        self.num_layers = np.log2(image_size) - 1
        self.latent_dim = latent_dim

        for layer in range(self.num_layers):
            self.generatorBlocks.append(GeneratorBlock(starting_filter, starting_filter * 2**, upsample = (layer == 0),
                                                       upsample_rgb=(layer == (range(self.num_layers )- 1))))
    def forward(self, latent_vectors):
        style_vectors = self.mappingNetwork(latent_vectors)
        rgb = None


class Discriminator(nn.Module):
    # gen filters/channels goes from image channels (3 for rgb) -> desired_features_discriminator * 1 ->
    # desired_features_discriminator * 2 -------> desired_features_discriminator * 8 (or something) -->
    # back to 1
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init()
        self.input_channels = input_channels
        self.output_channels = output_channels
        # skip connection for res blocks
        self.skip = nn.Conv2d(input_channels, output_channels, 1)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size)
    def forward(self, input):
        skipped_line_input = self.skip(input)
        main_line_input = self.conv1(input)
        main_line_input = self.conv2(input)
        return skipped_line_input + main_line_input

        

class StyleGan(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.generataor = Generator()
        self.discriminator = Discriminator()
        self.styleNetwork = MappingNetwork()
        self.GE = Generator()
        self.GE.load_state_dict(self.generator.state_dict())
        self.SE = Discriminator()
        self.SE.load_state_dict(self.styleNetwork.state_dict())
        self.generatorOptimizer = optim.Adam(self.generataor.parameters(), lr = 0.00001, betas = (0.5, 0.9))
        self.discriminatorOptimizer = optim.Adam(self.discriminator.paramaters(), lr = 0.00001, betas = (0, 0.9))

    # Estimated Moving Average
    def EMA(self, beta = 0.99):
        for S_params, SE_params, G_params, GE_params in zip(self.S.parameters(), self.SE.parameters(), self.G.parameters(), self.GE_parameters()):
            S_weights, SE_weights, G_weights, GE_weights= S_params.data, SE_params.data, G_params.data, GE_params.data
            if SE_weights is None:
                continue
            if GE_weights is None:
                continue
            SE_weights = SE_weights * beta + (1-beta) * S_weights
            GE_weights = GE_weights * beta + (1-beta) * G_weights

    def initializeMovingAverageWeights(self):
        self.SE.parameters().data = self.S.paramaters().data
        self.GE.parameters().data = self.G.parameters().data



class Trainer():
    def __init__(self, image_size, latent_dim,):
        self.StyleGan = StyleGan()
        self.image_size = image_size
        self.num_layers = np.log2(image_size)
        self.latent_dim = latent_dim
        assert image_size in [2**x for x in range(7, 11)]

    def train(self, batch_size, styles, epochs, steps,  mixed_probability=0.9,verbose=False):
        if np.random.random() < mixed_probability:
            noise = utils.createMixedNoise(batch_size, 512)
        else:
            noise = utils.createNoiseList(batch_size, 512)
        total_discriminator_loss = torch.tensor(0.).cuda()
        total_generator_loss = torch.tensor(0.).cuda()

        # training loop
        for epoch in epochs:
            for step in steps:
                w_space = []
                for style in range(styles):
                    w_space.append(self.styleNetwork(styles[style]))

                if steps % 16 == 0:
                    discriminator_loss += utils.gradientPenalty(data, images)

                if verbose == True:
                    if steps % 100 == 0:
                        print("Steps: ", steps)
                        print("Path Length Mean: ", )
                        print("Discriminator Loss: ", )
                        print("Generator Loss: ", )
                if steps > 20000:
                    self.EMA(0.99)
                if steps % 1000 == 0:
                    self.saveModel(steps)

    def evaluate(self):
        pass

    def saveModel(self, iteration):
        torch.save(self.StyleGan.state_dict(), f"Gan-{iteration}")

    def loadModel(self, iteration):
        print("Continuing from iteration", iteration, "!")
        self.steps = iteration
        self.StyleGan.load_state_dict(torch.load(f"Gan-{iteration}"))


