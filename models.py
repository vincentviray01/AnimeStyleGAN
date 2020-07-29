import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import torch.optim as optim
import numpy as np
from os import listdir, mkdir
import shutil
from torch.autograd import grad
import tensorboard

class Conv2DWeightDemod(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1,
                 dilation=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((output_channels, input_channels, kernel_size, kernel_size)))
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

    def get_same_padding(self, size, kernel, dilation, stride):
        return ((size -1 ) * (stride -1) + dilation * (kernel - 1)) // 2

    def forward(self, input_vector, style_vector):
        # style vector is of [num_layers, latent_dimensions]
        # Weight Demodulation
        style_vector = style_vector[:, None, :, None, None]
        weight = self.weight[None, :, :, :, :] * style_vector
        demodulation_coefficient = torch.rsqrt(weight.pow(2).sum([1, 2, 3], keepdims = True) + 1e-8)
        weight *= demodulation_coefficient
        print(style_vector.shape[0])
        print(style_vector.shape)
        print(input_vector.shape)
        print(weight.size())
        print(self.weight.size())
        print(self.output_channels * style_vector.shape[0])
        weight = weight.reshape(self.output_channels,self.input_channels, weight.shape[3], weight.shape[4])
        output = F.conv2d(input_vector, weight, padding = self.get_same_padding(input_vector.shape[3], self.kernel_size, dilation = self.dilation, stride = self.stride))
        print(output.size())
        print(self.output_channels)
        print(self.input_channels)
        print("WEE WOO WEE WOO")
        return nn.LeakyReLU(0.2)(output.view(-1, self.output_channels, input_vector.shape[2], input_vector.shape[3]))

class RGBBlock(nn.Module):
    def __init__(self, input_channels, latent_dim, rgb_upsample = True):
        # 3 into conv demod since RGB image has three channels
        super().__init__()
        self.rgb_upsample = rgb_upsample
        self.conv2dWeightDemod = Conv2DWeightDemod(input_channels, 3)
        # self.latent_to_style = nn.Linear(latent_dim, input_channels)

    def forward(self, image, prev_rgb, style_vector):
        # style_vector = self.latent_to_style(style_vector)
        image = self.conv2dWeightDemod(image, style_vector)
        if prev_rgb is not None:
            image = image + prev_rgb
        if self.rgb_upsample:
            image = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        return image

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, latent_dimensions = 512, kernel_size = 3, upsample = True, upsample_rgb = True, stride = 1, dilation = 1,
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
        self.RGBBlock = RGBBlock(input_channels, latent_dimensions, rgb_upsample=upsample_rgb)
        self.noise_to_channel = nn.Linear(1, output_channels)
        self.style_to_input_channels = nn.Linear(latent_dimensions, input_channels)

    def forward(self, image, prev_rgb, style_vector, noise: torch.Tensor):
        # style_vector is size (layers, latent_dimensions)
        # noise: [batch_size, image_size, image_size, 1] will be converted to [batch_size, image_size, image_size, output_channels]

        noise = noise[:, :image.shape[2], :image.shape[3], :]
        noise = self.noise_to_channel(noise).permute(0, 3, 1, 2)

        style_vector = self.style_to_input_channels(style_vector)
        print("style_vector is: ", style_vector.size())
        if self.upsample == True:
            image = nn.Upsample(scale_factor=2, mode='bilinear')(image)
        image = self.Conv2DWeightDemod(image, style_vector)
        image = nn.LeakyReLU(0.2)(image + noise)
        rgb = self.RGBBlock(image, prev_rgb, style_vector)
        return image, rgb



class DiscriminatorBlock(nn.Module):
    # will downsample using stride = 2
    def __init__(self, input_channels, output_channels, kernel_size = 3):
        super().__init__()
        # skip connection for res blocks
        self.skip = nn.Conv2d(int(input_channels), int(output_channels), kernel_size = 1, stride=2)
        # self.skip = nn.Conv2d(1, 1, 1, 1)

        self.mainLine = nn.Sequential(nn.Conv2d(int(input_channels), int(output_channels), kernel_size, padding = 1),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(int(output_channels), int(output_channels), kernel_size, padding = 1, stride=2))


    def forward(self, input):
        skipped_line_input = self.skip(input)
        main_line_input = self.mainLine(input)
        return nn.LeakyReLU(0.2)((skipped_line_input + main_line_input) / math.sqrt(2))


class MappingNetwork(nn.Module):
    def __init__(self, in_channels = 512, latent_channels = 512):
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
    def __init__(self, image_size, latent_dim, generator_filters, device):
        super().__init__()
        self.mappingNetwork = MappingNetwork(512, 512)
        self.start_of_network = False
        self.end_of_network = False
        self.generatorBlocks = []
        self.num_layers = int(np.log2(image_size) - 1)
        self.latent_dim = latent_dim
        # self.noise_to_first_image = nn.ConvTranspose2d(latent_dim, generator_filters, 4, 1, 0, bias = False)
        for layer in range(self.num_layers):
            if layer == 0:
                self.generatorBlocks.append(GeneratorBlock(generator_filters, generator_filters * 2**(self.num_layers - 1), latent_dimensions = latent_dim, upsample = (layer != 0),
                                                       upsample_rgb=(layer != (self.num_layers - 1))))
            else:
                self.generatorBlocks.append(
                    GeneratorBlock(generator_filters * 2**(self.num_layers - layer), generator_filters * (self.num_layers -1 - layer), latent_dimensions = latent_dim, upsample=(layer != 0),
                                   upsample_rgb=(layer != (self.num_layers - 1))))

        self.generatorBlocks = nn.Sequential(*self.generatorBlocks)
        self.initial_constant_input = torch.randn((1, generator_filters, 4, 4)).to(device)
    def forward(self, style_vector, input_noise):
        batch_size = input_noise.shape[0]
        x = self.initial_constant_input.expand(batch_size, -1, -1, -1)
        rgb = None
        style_vector = style_vector.transpose(0, 1)
        print("style vectors previous is: ", style_vector.size())
        for style, block in zip(style_vector, self.generatorBlocks):
            x, rgb = block(x, rgb, style, input_noise)
        return rgb






class Discriminator(nn.Module):
    # gen filters/channels goes from image channels (3 for rgb) -> desired_features_discriminator * 1 ->
    # desired_features_discriminator * 2 -------> desired_features_discriminator * 8 (or something) -->
    # back to 1
    def __init__(self, input_channels, discriminator_filters, latent_dim, image_size, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.num_layers = int(np.log2(image_size) - 1)

        # skip connection for res blocks
        self.discriminator_filters = discriminator_filters

        # The *2 * 2 is for the size of the image
        self.to_logit = nn.Linear((discriminator_filters * 2**(self.num_layers - 1)) * 2 * 2, 1)
        self.blocks = []
        for layer in range(self.num_layers):
            if layer == 0:
                self.blocks.append(DiscriminatorBlock(input_channels, discriminator_filters))
            else:
                self.blocks.append(DiscriminatorBlock(discriminator_filters * 2**(layer -1), discriminator_filters * 2**layer))
        self.discriminatorBlocks = nn.Sequential(*self.blocks)

    def forward(self, image):
        x = self.discriminatorBlocks(image)
        x = x.view(self.batch_size, -1)
        return self.to_logit(x)

        

class StyleGan(nn.Module):
    def __init__(self, batch_size, image_size, latent_dim, discriminator_filters, generator_filters, device):
        super().__init__()
        self.generator = Generator(image_size, latent_dim, generator_filters, device)
        self.discriminator = Discriminator(3, discriminator_filters, latent_dim, image_size, batch_size)
        self.styleNetwork = MappingNetwork(latent_dim, latent_dim)
        # self.GE = Generator()
        # self.GE.load_state_dict(self.generator.state_dict())
        # self.SE = Discriminator()
        # self.SE.load_state_dict(self.styleNetwork.state_dict())
        self.generatorOptimizer = optim.Adam(self.generator.parameters(), lr = 0.00001, betas = (0.5, 0.9))
        self.discriminatorOptimizer = optim.Adam(self.discriminator.parameters(), lr = 0.00001, betas = (0, 0.9))
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

    def init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')



class Trainer():
    def __init__(self, batch_size, image_size, latent_dim, epochs, discriminator_filters, generator_filters, device, mixed_probability = 0.9, pl_beta = 0.99):
        self.StyleGan = StyleGan(batch_size, image_size, latent_dim, discriminator_filters, generator_filters, device).to(device)
        self.image_size = image_size
        self.num_layers = np.log2(image_size)
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        assert image_size in [2**x for x in range(7, 11)]
        self.discriminator_loss = torch.tensor(0.).to(device)
        self.generator_loss = torch.tensor(0.).to(device)
        self.dataLoader = utils.getDataLoader(batch_size, image_size)
        self.mixed_probability = mixed_probability
        self.epochs = epochs
        self.loss_fn = nn.BCELoss()
        self.average_pl_length = 0
        self.pl_beta = pl_beta
        self.device = device


    def train(self, verbose=False):
        # load last iteration if training was started but not finished
        if len(listdir("saves")) > 0:
            self.StyleGan.load_state_dict(torch.load("saves/" + listdir("saves")[-1]))
        else:
            self.StyleGan.init_weights()

        self.StyleGan.generator.train()
        self.StyleGan.discriminator.train()
        if np.random.random() < self.mixed_probability:
            style_noise = utils.createStyleMixedNoiseList(self.batch_size, self.latent_dim, self.num_layers, self.StyleGan.styleNetwork, self.device)
        else:
            style_noise = utils.createStyleNoiseList(self.batch_size, self.latent_dim, self.num_layers, self.StyleGan.styleNetwork, self.device)
        image_noise = utils.create_image_noise(self.batch_size, self.image_size, self.device)

        # utils.init_weights(self.StyleGan)
        # utils.set_requires_grad(self.StyleGan, True)

        # training loop
        for epoch in range(self.epochs):
            self.saveModel(epoch)
            for batch_num, batch in enumerate(self.dataLoader):
                batch = batch[0].to(self.device)
                batch.requires_grad = True

                w_space = []
                # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

                self.StyleGan.discriminator.zero_grad()
                real_labels = (torch.ones(self.batch_size) * 0.9).to(self.device)

                # transpose to match size with label tensor size
                discriminator_real_output = torch.transpose(self.StyleGan.discriminator(batch), 0, 1).to(self.device)
                print(real_labels)
                print("HFUSDHF")
                print(discriminator_real_output)
                print(discriminator_real_output.size())

                discriminator_real_loss = self.loss_fn(discriminator_real_output, real_labels)
                discriminator_average_real_loss = discriminator_real_loss.mean().item()

                generated_images = self.StyleGan.generator(style_noise, image_noise).to(self.device)
                fake_labels = (torch.ones(self.batch_size) * 0.1).to(self.device)

                # transpose to match size with label tensor size
                discriminator_fake_output = torch.transpose(self.StyleGan.discriminator(generated_images.detatch()), 0, 1).to(self.device)
                discriminator_fake_loss = self.loss_fn(discriminator_fake_output, fake_labels).to(self.device)
                discriminator_average_fake_loss = discriminator_fake_loss.mean().item().to(self.device)

                discriminator_total_loss = discriminator_fake_loss + discriminator_real_loss
                # discriminator_accuracy = 0

                # Apply Gradient Penalty every 4 steps
                if batch_num % 4 == 0:
                    discriminator_total_loss += utils.gradientPenalty(batch, discriminator_real_output, self.device).detach().item().to(self.device)

                discriminator_total_loss.backward()
                self.StyleGan.discriminatorOptimizer.step()

                # Train Generator: maximize log(D(G(z)))
                self.StyleGan.generator.zero_grad()
                generator_labels = torch.ones(self.batch_size).to(self.device)
                generator_output = self.StyleGan.discriminator(generated_images).reshape(-1).to(self.device)
                generator_loss = self.loss_fn(generator_labels, generator_output).to(self.device)


                # Apply Path Length Regularization every 16 steps
                if batch_num % 16 == 0:
                    num_pixels = generated_images.shape[2] * generated_images[3].to(self.device)
                    noise_to_add = torch.randn(generated_images.shape).cuda() / math.sqrt(num_pixels).to(self.device)
                    outputs = (generated_images * noise_to_add())

                    pl_gradient = grad(outputs = outputs, inputs = style_noise, grad_outputs = torch.ones(outputs.shape).cuda(),
                                       create_graph=True, retain_graph=True, only_inputs=True)[0].to(self.device)
                    pl_length = torch.sqrt(torch.sum(torch.square(pl_gradient)))



                    pl_regularizer = (pl_length - self.average_pl_length).mean()

                    if self.average_pl_length == 0:
                        self.average_pl_length == pl_length

                    generator_loss += pl_regularizer

                #Update average path length
                self.average_pl_length = self.average_pl_length * self.pl_beta + (1-self.pl_beta) * pl_length

                generator_loss.backward()
                generator_accuracy = generator_loss.argmax == generator_labels  # TODO
                self.StyleGan.generatorOptimizer.step()

                if verbose == True:
                    if batch_num % 100 == 0:
                        print("Batch: ", batch_num)
                        print("Epoch: ", epoch)
                        print("Path Length Mean: ", )
                        print("Discriminator Mean Real Loss: ", discriminator_average_real_loss)
                        print("Discriminator Mean Fake Loss: ", discriminator_average_fake_loss)
                        # print("Discriminator Accuracy: ", discriminator_accuracy)
                        print("Generator Loss: ", generator_loss )
                        # print("Generator Accuracy: ", generator_accuracy)
                # if steps > 20000:
                #     self.StyleGan.EMA(0.99)


    @torch.no_grad()
    def evaluate(self):
        pass

    def create_interpolation(self):
        pass

    def saveModel(self, iteration):
        torch.save(self.StyleGan.state_dict(), f"saves/Gan-{iteration}.pt")

    def loadModel(self, iteration):
        print("Continuing from iteration", iteration, "!")
        self.StyleGan.load_state_dict(torch.load(f"saves/Gan-{iteration}".pt))

    def resetSaves(self):
        shutil.rmtree('saves')
        mkdir("saves")