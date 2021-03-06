import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import torch.optim as optim
import numpy as np
from os import listdir, mkdir
import shutil
from torch.autograd import grad, set_detect_anomaly
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from math import isnan
from matplotlib import pyplot as plt

try:
    from apex import amp
    apex_available = True
    # amp.register_float_function(torch, 'sigmoid')
except ModuleNotFoundError:
    apex_available = False
from tqdm import tqdm

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
        b, *_ = input_vector.shape

        # clone() is used to create a separate copy in order to not modify the original with inplace operations
        input_vector = input_vector.reshape(1, -1, input_vector.shape[2], input_vector.shape[3])
        style_vector_new = style_vector[:, None, :, None, None]
        weight = self.weight[None, :, :, :, :]
        weights = weight.clone() * style_vector_new
        demodulation_coefficient = torch.rsqrt((weights.pow(2)).sum([1, 2, 3], keepdims = True) + 1e-8)
        weights = weights.clone() * demodulation_coefficient

        weights = weights.reshape(self.output_channels * weights.shape[0],self.input_channels, weights.shape[3], weights.shape[4])
        output = F.conv2d(input_vector, weights, padding = self.get_same_padding(input_vector.shape[3], self.kernel_size, dilation = self.dilation, stride = self.stride), groups = b)
        return nn.LeakyReLU(0.2)(output.view(-1, self.output_channels, input_vector.shape[2], input_vector.shape[3]))

class RGBBlock(nn.Module):
    def __init__(self, input_channels, latent_dim, rgb_upsample = True):
        # 3 into conv demod since RGB image has three channels
        super().__init__()
        self.rgb_upsample = rgb_upsample
        self.conv2dWeightDemod = Conv2DWeightDemod(input_channels, 3)
        self.latent_to_style = nn.Linear(latent_dim, input_channels)


    def forward(self, image, prev_rgb, style_vector):
        style_vector_new = self.latent_to_style(style_vector)
        image = self.conv2dWeightDemod(image, style_vector_new)
        if prev_rgb is not None:
            image = image + prev_rgb
        if self.rgb_upsample:
            image = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(image)

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
        self.RGBBlock = RGBBlock(output_channels, latent_dimensions, rgb_upsample=upsample_rgb)
        self.noise_to_channel = nn.Linear(1, output_channels)
        self.style_to_input_channels = nn.Linear(latent_dimensions, input_channels)

    def forward(self, image, prev_rgb, style_vector, noise: torch.Tensor):
        # style_vector is size (layers, latent_dimensions)
        # noise: [batch_size, image_size, image_size, 1] will be converted to [batch_size, image_size, image_size, output_channels]


        style_vector_new = self.style_to_input_channels(style_vector)
        if self.upsample == True:
            image = nn.Upsample(scale_factor=2, mode='bilinear')(image)
        noise = noise[:, :image.shape[2], :image.shape[3], :]
        noise = self.noise_to_channel(noise).permute(0, 3, 1, 2)
        image = self.Conv2DWeightDemod(image, style_vector_new)
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
        # self.mappingNetwork = MappingNetwork(512, 512)
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
                    GeneratorBlock(generator_filters * 2**(self.num_layers - layer), generator_filters * 2**(self.num_layers -1 - layer), latent_dimensions = latent_dim, upsample=(layer != 0),
                                   upsample_rgb=(layer != (self.num_layers - 1))))

        self.generatorBlocks = nn.Sequential(*self.generatorBlocks)
        self.initial_constant_input = torch.randn((1, generator_filters, 4, 4)).to(device)
    def forward(self, style_vector, input_noise):
        batch_size = input_noise.shape[0]
        x = self.initial_constant_input.expand(batch_size, -1, -1, -1).half()
        rgb = None
        style_vector = style_vector.transpose(0, 1)
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
        self.generator = Generator(image_size, latent_dim, generator_filters, device).to(device)
        self.discriminator = Discriminator(3, discriminator_filters, latent_dim, image_size, batch_size).to(device)
        self.styleNetwork = MappingNetwork(latent_dim, latent_dim).to(device)
        # self.GE = Generator()
        # self.GE.load_state_dict(self.generator.state_dict())
        # self.SE = Discriminator()
        # self.SE.load_state_dict(self.styleNetwork.state_dict())
        self.generatorOptimizer = optim.Adam(self.generator.parameters(), lr = 0.001, betas = (0.5, 0.99))
        self.discriminatorOptimizer = optim.Adam(self.discriminator.parameters(), lr = 0.004, betas = (0.5, 0.99))
        # self.styleNetworkOptimizer = optim.Adam(self.styleNetwork.parameters(), lr = 0.001, betas = (0, 0.9))
        self.generator, self.generatorOptimizer = amp.initialize(self.generator, self.generatorOptimizer, opt_level="O2",
                                                                 keep_batchnorm_fp32=None, loss_scale="dynamic", max_loss_scale=2**13)
        self.discriminator, self.discriminatorOptimizer = amp.initialize(self.discriminator, self.discriminatorOptimizer,
                                                                 opt_level="O2",
                                                                 keep_batchnorm_fp32=None, loss_scale="dynamic", max_loss_scale=2**13)
        self.styleNetwork = amp.initialize(self.styleNetwork, opt_level="O2", keep_batchnorm_fp32=None, loss_scale="dynamic", max_loss_scale=2**13)

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
    def __init__(self, batch_size, image_size, latent_dim, epochs, discriminator_filters, generator_filters, device, mixed_probability = 0.9, pl_beta = 0.9):
        self.StyleGan = StyleGan(batch_size, image_size, latent_dim, discriminator_filters, generator_filters, device).to(device)
        self.image_size = image_size
        self.num_layers = np.log2(image_size)
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        assert image_size in [2**x for x in range(5, 11)]
        self.discriminator_loss = torch.tensor(0.).to(device)
        self.generator_loss = torch.tensor(0.).to(device)
        self.dataLoader = utils.getDataLoader(batch_size, image_size)
        self.mixed_probability = mixed_probability
        self.epochs = epochs
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.average_pl_length = None
        self.pl_beta = pl_beta
        self.device = device
        self.tensorboard_summary = SummaryWriter('runs7/stylegan2')
        self.checkpoint = 0
        self.apex_available = apex_available
        self.constant_style = utils.createStyleMixedNoiseList(self.batch_size, self.latent_dim, self.num_layers,
                                                                  self.StyleGan.styleNetwork, self.device)
        self.constant_noise = utils.create_image_noise(self.batch_size, self.image_size, self.device)

    def train(self, verbose=True):

        # set_detect_anomaly(True)
        torch.cuda.empty_cache()

        # load last iteration if training was started but not finished
        if len(listdir("saves7")) > 0:
            # the [4::-3] is because of the file name format, with the number of each checkpoint at these points
            self.loadModel(sorted(listdir('saves7'), key = lambda x: int(x[4: -3]))[-1][4:-3])
            # self.loadModel(120)
            self.checkpoint = int(sorted(listdir('saves7'), key = lambda x: int(x[4: -3]))[-1][4:-3])
            print("Loading from checkpoint: ", self.checkpoint)
            self.checkpoint = self.checkpoint + 1
            print("New checkpoint starts at: ", self.checkpoint)
        else:
            self.StyleGan.init_weights()
            self.checkpoint = 0
        # self.loadModel(1050)
        # self.checkpoint = 0
        # print(self.StyleGan)

        # utils.init_weights(self.StyleGan)
        # utils.set_requires_grad(self.StyleGan, True)

        # training loop
        for epoch in range(0, self.epochs):
            for batch_num, batch in enumerate(self.dataLoader):

                # if batch_num % 50 == 0:
                #     # generated_images = self.StyleGan.generator(self.constant_style, self.constant_noise)
                #     img_grid = make_grid(generated_images)
                #     self.tensorboard_summary.add_image(f'generated_image{self.checkpoint}', img_grid)
                #     del generated_images
                #     del img_grid

                batch = batch[0].expand(-1, 3, -1, -1).to(self.device)
                batch.requires_grad = True

                # print("OMG", batch.shape)
                if batch.shape[0] != 128:
                    print("SKIPPING")
                    continue

                w_space = []
                # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                utils.set_requires_grad(self.StyleGan.discriminator, True)
                self.StyleGan.discriminator.train()
                self.StyleGan.generator.eval()
                utils.set_requires_grad(self.StyleGan.generator, False)

                if np.random.random() < self.mixed_probability:
                    style_noise = utils.createStyleMixedNoiseList(self.batch_size, self.latent_dim, self.num_layers,
                                                                  self.StyleGan.styleNetwork, self.device)
                else:
                    style_noise = utils.createStyleNoiseList(self.batch_size, self.latent_dim, self.num_layers,
                                                             self.StyleGan.styleNetwork, self.device)
                image_noise = utils.create_image_noise(self.batch_size, self.image_size, self.device)

                # style_noise = style_noise.half()
                # image_noise = image_noise.half()

                self.StyleGan.discriminatorOptimizer.zero_grad()
                real_labels = (torch.ones(self.batch_size) * 0.9).to(self.device)

                # transpose to match size with label tensor size
                discriminator_real_output = self.StyleGan.discriminator(batch).reshape(-1).to(self.device)
                # print("DIS real output", discriminator_real_output)

                discriminator_real_loss = self.loss_fn(discriminator_real_output, real_labels).mean()
                # print("DIS real loss", discriminator_real_loss)
                # print("DIS real labelsL", real_labels)

                del real_labels

                generated_images = self.StyleGan.generator(style_noise.detach(), image_noise.detach()).to(self.device)
                del style_noise
                del image_noise
                fake_labels = (torch.ones(self.batch_size) * 0.1).to(self.device)

                # transpose to match size with label tensor size
                discriminator_fake_output = self.StyleGan.discriminator(generated_images.detach()).reshape(-1).to(self.device)
                # print("DIS fake output", discriminator_fake_output)
                discriminator_fake_loss = self.loss_fn(discriminator_fake_output, fake_labels).mean()
                # print("DIS fake loss", discriminator_fake_loss)
                # print("DIS fake labels", fake_labels)

                del fake_labels
                del discriminator_fake_output


                discriminator_total_loss = discriminator_fake_loss + discriminator_real_loss
                # if batch_num % 100 == 0:
                #     print("d real loss", discriminator_real_loss)
                #     print("d fake loss", discriminator_fake_loss)
                #     print("d real output", discriminator_real_output)
                #     print("d fake output", discriminator_fake_output)
                #     print("real + fake loss", discriminator_fake_loss + discriminator_real_loss)
                #     print("total loss", discriminator_total_loss)
                #     print("\n\n")


                # discriminator_accuracy = 0

                # Apply Gradient Penalty every 4 steps
                # if batch_num % 4 == 0:
                # print("before gradien tpenalty", batch_num)
                discriminator_total_loss = discriminator_total_loss + utils.gradientPenalty(batch, discriminator_real_output, self.device)

                del discriminator_real_output

                if isnan(discriminator_total_loss):
                    print("IS NAN discriminator")
                    break


                if self.apex_available:
                    with amp.scale_loss(discriminator_total_loss, self.StyleGan.discriminatorOptimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    discriminator_total_loss.backward()


                torch.nn.utils.clip_grad_norm_(self.StyleGan.discriminator.parameters(), 5, norm_type=2)
                # for p in self.StyleGan.discriminator.parameters():
                #     p.data.clamp_(-0.01, 0.01)

                self.StyleGan.discriminatorOptimizer.step()
                # Train Generator: maximize log(D(G(z)))

                utils.set_requires_grad(self.StyleGan.discriminator, False)
                self.StyleGan.discriminator.eval()
                self.StyleGan.generator.train()
                utils.set_requires_grad(self.StyleGan.generator, True)
                if np.random.random() < self.mixed_probability:
                    style_noise = utils.createStyleMixedNoiseList(self.batch_size, self.latent_dim, self.num_layers,
                                                                  self.StyleGan.styleNetwork, self.device)
                else:
                    style_noise = utils.createStyleNoiseList(self.batch_size, self.latent_dim, self.num_layers,
                                                             self.StyleGan.styleNetwork, self.device)
                image_noise = utils.create_image_noise(self.batch_size, self.image_size, self.device)

                self.StyleGan.generatorOptimizer.zero_grad()
                generator_labels = torch.ones(self.batch_size).to(self.device)

                # Generate images
                generated_images = self.StyleGan.generator(style_noise, image_noise).to(self.device)
                del image_noise

                # utils.showImage(batch)
                # utils.showImage(generated_images)



                generator_output = self.StyleGan.discriminator(generated_images).reshape(-1).to(self.device)
                # print("gen output", generator_output)
                generator_loss = self.loss_fn(generator_output, generator_labels).mean()
                # print("gen loss", generator_loss)
                # print("gen labels", generator_labels)


                # if batch_num % 100 == 0:
                #     print(generator_loss)

                if isnan(generator_loss):
                    print("isnan generator")
                    break

                del generator_output
                del generator_labels

                generator_loss_no_pl = generator_loss

                # Apply Path Length Regularization every 16 steps
                if batch_num % 10 == 0:
                    num_pixels = generated_images.shape[2] * generated_images.shape[3]
                    noise_to_add = (torch.randn(generated_images.shape)/ math.sqrt(num_pixels)).to(self.device)
                    outputs = (generated_images * noise_to_add)

                    # del generated_images
                    pl_gradient = grad(outputs = outputs,
                                       inputs = style_noise, grad_outputs = torch.ones(outputs.shape).to(self.device),
                                       create_graph=True, retain_graph=True, only_inputs=True)[0]
                    del num_pixels
                    del noise_to_add
                    del outputs

                    pl_length = torch.sqrt(torch.sum(torch.square(pl_gradient)))


                    if self.average_pl_length is not None:
                        pl_regularizer = ((pl_length - self.average_pl_length)**2).mean()
                    else:
                        pl_regularizer = (pl_length**2).mean()


                    del pl_gradient

                    del style_noise


                    # print("PL LENGTH IS: ", pl_length)
                    if self.average_pl_length == None:
                        self.average_pl_length = pl_length.detach().item()
                    else:
                        self.average_pl_length = self.average_pl_length * self.pl_beta + (1 - self.pl_beta) * pl_length.detach().item()
                    # self.average_pl_length = pl_length

                    del pl_length

                    generator_loss = generator_loss + pl_regularizer

                    # print(self.average_pl_length)
                    # print(batch_num)




                # print(batch_num, "HI")
                if self.apex_available:
                    with amp.scale_loss(generator_loss, self.StyleGan.generatorOptimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    generator_loss.backward(retain_graph = True)


                torch.nn.utils.clip_grad_norm_(self.StyleGan.generator.parameters(), 5, norm_type=2)
                # for p in self.StyleGan.generator.parameters():
                #     p.data.clamp_(-0.01, 0.01)
                # generator_accuracy = generator_loss.argmax == generator_labels  # TODO
                self.StyleGan.generatorOptimizer.step()

                # Update MappingNetwork weights
                # if self.apex_available:
                #     with amp.scale_loss(generator_loss, self.StyleGan.generatorOptimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     generator_loss.backward(retain_graph = True)

                if verbose == True:
                    if batch_num % 100 == 0 and batch_num != 0:
                        # print("average path length is: ", self.average_pl_length)
                        print("Checkpoint")
                        print("Batch: ", batch_num)
                        print("Path Length Mean: ", self.average_pl_length)
                        print("Discriminator Mean Real Loss: ", discriminator_real_loss.item())
                        print("Discriminator Mean Fake Loss: ", discriminator_fake_loss.item())
                        print("Discriminator Total Loss: ", discriminator_total_loss.item())
                        # print("Discriminator Accuracy: ", discriminator_accuracy)
                        print("Generator Loss (no pl)", generator_loss_no_pl)
                        print("Generator Loss: ", generator_loss.item())
                        # print("Generator Accuracy: ", generator_accuracy)
                        print("PL difference:", pl_regularizer.item())

                        
                if batch_num % 100 == 0 and batch_num != 0:
                    print("Current Checkpoint is: ", self.checkpoint)
                    img_grid = make_grid(generated_images)

                    self.tensorboard_summary.add_scalar('Path Length Mean', self.average_pl_length, self.checkpoint)
                    self.tensorboard_summary.add_scalar('Discriminator Mean Real Loss ',
                                                        discriminator_real_loss, self.checkpoint)
                    self.tensorboard_summary.add_scalar('Discriminator Mean Fake Loss ',
                                                        discriminator_fake_loss, self.checkpoint)
                    self.tensorboard_summary.add_scalar('Discriminator Total Loss ', discriminator_total_loss.item(),
                                                        self.checkpoint)
                    self.tensorboard_summary.add_scalar('Generator Loss', generator_loss.item(), self.checkpoint)
                    self.tensorboard_summary.add_scalar('Path Length Difference', pl_regularizer.item(), self.checkpoint)
                    self.tensorboard_summary.add_scalar('Generator Loss (No PL)', generator_loss_no_pl.item(), self.checkpoint)
                    self.tensorboard_summary.add_image(f'generated_image{self.checkpoint}', img_grid)
                    # self.tensorboard_summary.add_scalar("D")
                    del generated_images
                    del img_grid
                    # self.tensorboard_summary.add_scalar('Generator Weight', self.StyleGan.generator.we, self.checkpoint)
                    # self.tensorboard_summary.add_scalar('Generator Weight', generator_loss_no_pl.item(), self.checkpoint)
                    del generator_loss_no_pl
                    del discriminator_total_loss
                    del generator_loss
                    del pl_regularizer
                    del discriminator_real_loss
                    del discriminator_fake_loss
                    self.saveModel(self.checkpoint)
                    self.checkpoint = self.checkpoint + 1



                # if steps > 20000:
                #     self.StyleGan.EMA(0.99)

            # Right now, an epoch is never achieved
            # # Create a checkpoint at the end of an epoch
            # print("Current Checkpoint is: ", self.checkpoint)
            # self.tensorboard_summary.add_scalar('Path Length Mean', self.average_pl_length, self.checkpoint)
            # self.tensorboard_summary.add_scalar('Discriminator Mean Real Loss ',
            #                                     discriminator_real_loss, self.checkpoint)
            # self.tensorboard_summary.add_scalar('Discriminator Mean Fake Loss ',
            #                                     discriminator_fake_loss, self.checkpoint)
            # self.tensorboard_summary.add_scalar('Discriminator Total Loss ', discriminator_total_loss.item(),
            #                                     self.checkpoint)
            # self.tensorboard_summary.add_scalar('Generator Loss', generator_loss.item(), self.checkpoint)
            # self.tensorboard_summary.add_scalar('Path Length Difference', pl_regularizer.item(), self.checkpoint)
            # self.tensorboard_summary.add_scalar('Generator Loss (No PL)', generator_loss_no_pl.item())
            # del generator_loss_no_pl
            # del discriminator_total_loss
            # del generator_loss
            # del pl_regularizer
            # del discriminator_fake_loss
            # del discriminator_real_loss
            # self.saveModel(self.checkpoint)
            # self.checkpoint = self.checkpoint + 1



        # Close TensorBoard at the end
        self.tensorboard_summary.close()


    @torch.no_grad()
    def evaluate(self):

        def show(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

        # load last iteration if training was started but not finished
        if len(listdir("saves7")) > 0:
            # the [4::-3] is because of the file name format, with the number of each checkpoint at these points
            self.loadModel(27)
        else:
            self.StyleGan.init_weights()
            self.checkpoint = 0

        if np.random.random() < self.mixed_probability:
            style_noise = utils.createStyleMixedNoiseList(self.batch_size, self.latent_dim, self.num_layers,
                                                          self.StyleGan.styleNetwork, self.device)
        else:
            style_noise = utils.createStyleNoiseList(self.batch_size, self.latent_dim, self.num_layers,
                                                     self.StyleGan.styleNetwork, self.device)
        image_noise = utils.create_image_noise(self.batch_size, self.image_size, self.device)
        generated_images = self.StyleGan.generator(style_noise, image_noise)
        for x in range(generated_images.shape[0]):
            utils.showImage(generated_images[x])
        # img_grid = make_grid(generated_images[0])
        # show(img_grid.cpu())

    def create_interpolation(self):
        pass

    def saveModel(self, iteration):
        save_dict = {'generatorModel': self.StyleGan.generator.state_dict(),
                     'generatorModelOptimizer': self.StyleGan.generatorOptimizer.state_dict(),
                     "discriminatorModel": self.StyleGan.discriminator.state_dict(),
                     "discriminatorModelOptimizer": self.StyleGan.discriminatorOptimizer.state_dict(),
                     'amp': amp.state_dict(),
                     "average_pl": self.average_pl_length,
                     "constant_style": self.constant_style,
                     "constant_noise": self.constant_noise,
                     "style_network": self.StyleGan.styleNetwork.state_dict(),
                     "constant_input": self.StyleGan.generator.initial_constant_input}

        torch.save(save_dict, f"saves7/Gan-{iteration}.pt")

    def loadModel(self, iteration):
        load_dict = torch.load(f"saves7/Gan-{iteration}.pt")
        # print(load_dict)
        #
        # load_dict["generatorModelOptimizer"]["param_groups"][0]['lr'] = 0.001
        # load_dict["discriminatorModelOptimizer"]["param_groups"][0]['lr'] = 0.001

        # load_dict["generatorModelOptimizer"]["param_groups"][0]['betas'] = (0.5, 0.99)
        # print(load_dict["average_pl"])
        self.StyleGan.generator.load_state_dict(load_dict["generatorModel"])
        self.StyleGan.generatorOptimizer.load_state_dict(load_dict["generatorModelOptimizer"])
        self.StyleGan.discriminator.load_state_dict(load_dict["discriminatorModel"])
        self.StyleGan.discriminatorOptimizer.load_state_dict(load_dict["discriminatorModelOptimizer"])
        self.average_pl_length = load_dict["average_pl"]
        self.constant_style = load_dict["constant_style"]
        self.constant_noise = load_dict["constant_noise"]
        self.StyleGan.styleNetwork.load_state_dict(load_dict["style_network"])
        self.StyleGan.generator.initial_constant_input = load_dict["constant_input"]
        amp.load_state_dict(load_dict["amp"])

    def resetSaves(self):
        shutil.rmtree('saves')
        mkdir("saves")
        shutil.rmtree('runs')
        mkdir("runs")