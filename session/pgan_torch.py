import glob
import yaml
from yaml import Loader, Dumper
from itertools import chain
from unet_segmentor_lib import *
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import functools
import numpy as np

C_AXIS = 1


class DimsTransformer:
    def __init__(self, batch_transform=None, size_transform=None, channels_transform=None, c_axis=C_AXIS):
        self.c_axis = c_axis
        self.size_transform = size_transform
        self.channels_transform = channels_transform
        self.batch_transform = batch_transform

    def __call__(self, dims):
        channels = dims[self.c_axis] if self.channels_transform is None else self.channels_transform(dims[self.c_axis])
        batch_size = dims[0] if self.batch_transform is None else self.batch_transform(dims[0])
        other_axes = (ax for ax in range(len(dims)) if ax not in [0, self.c_axis])
        size = dims[other_axes] if self.size_transform is None else self.size_transform(dims[other_axes])
        if self.c_axis == 1:
            new_size = [batch_size, channels, size]
        else:
            new_size = [batch_size, size, channels]
        return new_size


class Conv2dTransformer(DimsTransformer):
    def __init__(self, filters, kernel_size, stride, padding="same", dilation=(1, 1)):
        def size_transform(input_size, pad=padding):
            if pad == "same":
                if stride != (1, 1):
                    raise ValueError("conv cannot have same padding with stride > 1")
                return input_size
            elif pad == "valid":
                pad = (0, 0)

            def output_size(in_size: int, ax:int, p=pad):
                out_size = in_size + 2 * p[ax] - dilation[ax] * (kernel_size[ax] - 1) - 1
                out_size = out_size / stride[ax]
                out_size += 1
                out_size = int(out_size)
                return out_size

            return [output_size(ax, i) for i, ax in enumerate(input_size)]

        def channels_transform(input_filters):
            return filters

        super(Conv2dTransformer, self).__init__(size_transform=size_transform, channels_transform=channels_transform,
                                                c_axis=C_AXIS)


class ConvTrans2dTransformer(DimsTransformer):
    def __init__(self, filters, kernel_size, stride,
                 padding=(0, 0), dilation=(1, 1), output_padding=(0, 0)):
        def size_transform(input_size, pad=padding):
            if pad == "same":
                if stride != (1, 1):
                    raise ValueError("conv cannot have same padding with stride > 1")
                return input_size
            elif pad == "valid":
                pad = (0, 0)

            def output_size(in_size: int, ax:int, p=pad):
                out_size = (in_size - 1) * stride[ax] - 2 * p[ax] + \
                           dilation[ax] * (kernel_size[ax] - 1) + output_padding[ax] + 1
                return out_size

            return [output_size(ax, i) for i, ax in enumerate(input_size)]

        def channels_transform(input_filters):
            return filters

        super(ConvTrans2dTransformer, self).__init__(size_transform=size_transform,
                                                     channels_transform=channels_transform, c_axis=C_AXIS)


class DimsTracker:
    def __init__(self, input_dim, c_axis=C_AXIS):
        self.curr_dim = input_dim
        self.prev_dim = input_dim
        self.c_axis = c_axis

    def update_dims(self, transform):
        self.prev_dim = self.curr_dim
        self.curr_dim = transform(self.curr_dim)

    def get_dims(self):
        return self.prev_dim, self.curr_dim

    def curr_dim(self):
        return self.curr_dim()

    def prev_dim(self):
        return self.prev_dim

    def prev_channels(self):
        return self.prev_dim[self.c_axis]

    def curr_channels(self):
        return self.curr_dim[self.c_axis]

    def curr_size(self):
        return [dim for i, dim in enumerate(self.curr_dim) if i not in [0, self.c_axis]]

    def prev_size(self):
        return [dim for i, dim in enumerate(self.prev_dim) if i not in [0, self.c_axis]]

    def num_prev_features(self):
        return functools.reduce(lambda x, y: x+y, self.prev_dim())

    def num_curr_features(self):
        return functools.reduce(lambda x, y: x+y, self.curr_dim())


def new_conv2d(curr_dims: DimsTracker, filters, kernel_size=(3, 3), stride=(1, 1), padding="same", dilation=(1, 1)):
    transform = Conv2dTransformer(filters, kernel_size=kernel_size, padding=padding, stride=stride)
    conv = torch.nn.Conv2d(curr_dims.get_channels()[0], filters, kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation)
    curr_dims.update_dims(transform)
    return conv, curr_dims


def new_conv_trans2d(curr_dims: DimsTracker, filters, kernel_size=(3, 3), stride=(1, 1),
                     padding=(0, 0), dilation=1):
    transform = ConvTrans2dTransformer(filters, kernel_size=kernel_size, padding=padding, stride=stride)
    conv = torch.nn.ConvTranspose2d(curr_dims.get_channels()[0], filters, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation)
    curr_dims.update_dims(transform)
    return conv, curr_dims


def new_linear(curr_dims: DimsTracker, units):
    curr_dims.update_dims(transform=lambda _: [curr_dims.curr_dim()[0], units])
    linear = torch.nn.Linear(in_features=curr_dims.num_prev_features(), out_features=curr_dims.num_curr_features())
    return linear, curr_dims


class EncoderBlock(torch.nn.Module):
    def __init__(self, input_dim, filters):
        super(EncoderBlock, self).__init__()
        dims = DimsTracker(input_dim)
        self.conv1, dims = new_conv2d(dims, filters=filters)
        self.ln1 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim()[1:])
        self.conv2, dims = new_conv2d(dims, filters=filters)
        self.ln2 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim()[1:])
        # cat here
        dims.update_dims(transform=DimsTransformer(channels_transform=lambda x: 2 * x))
        self.conv3, dims = new_conv2d(dims, filters=filters, stride=(2, 2), padding=(1, 1))  # output image size halved
        self.ln3 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim()[1:])

        self.out_dims = dims

    def forward(self, inputs):
        x = self.conv1(inputs)  # output_dim_pre_concat
        x = torch.nn.LeakyReLU(negative_slope=0.1)(x)  # intermediate_dim
        x = self.ln1(x)  # intermediate_dim
        x = self.conv2(x)  # intermediate_dim
        x = torch.nn.LeakyReLU(negative_slope=0.1)(x)  # intermediate_dim
        x = self.ln2(x)  # intermediate_dim
        x = torch.cat([x, inputs], dim=C_AXIS)  # skip connection, C_AXIS: filters * 2
        x = self.conv3(x)  # output_dim
        x = torch.nn.LeakyReLU(negative_slope=0.1)(x)  # output_dim
        x = self.ln3(x)  # output_dim
        return x


class GeneratorBlock(torch.nn.Module):
    def __init__(self, input_dim, filters):
        super(GeneratorBlock, self).__init__()
        dims = DimsTracker(input_dim)
        self.conv1, dims = new_conv_trans2d(dims, filters=filters, kernel_size=(3, 3),
                                            stride=(2, 2), padding=(1, 1))  # doubles input size
        self.ln1 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim()[1:])
        self.conv2, dims = new_conv2d(filters=filters)
        self.ln2 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim()[1:])
        self.conv3, dims = new_conv2d(filters=filters)
        self.ln3 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim()[1:])

        self.out_dims = dims

    def forward(self, inputs):
        x = self.conv1(inputs)
        y = torch.nn.ReLU()(x)
        y = self.ln1(y)
        y = self.conv2(y)
        y = torch.nn.ReLU()(y)
        y = self.ln2(y)
        y = self.conv3(y)
        y = torch.nn.ReLU()(y)
        y = self.ln3(y)
        y = x + y  # skip connection
        return y


def to_image_tanh(x):
    return torch.tanh(x) - 1


class ShowGeneratorCallback(keras.callbacks.Callback):
    def __init__(self, display=100):
        super(ShowGeneratorCallback, self).__init__()
        self.seen = 0
        self.display = display

    def print_pics(self):
        if self.seen % self.display > 0:
            return

        batch_size = tf.constant(9)
        code_shape = self.model.code_shape
        code_features = self.model.code_features
        noise_features = self.model.noise_features

        latent_shape = (batch_size, code_shape[1], code_shape[2],
                        code_features + noise_features)
        random_latent_vectors = tf.random.normal(shape=latent_shape)

        alpha = self.model.current_alpha
        alpha = tf.constant(alpha, dtype=tf.float32)
        random_pics = self.model.generator([random_latent_vectors, alpha * tf.ones([batch_size])])

        plt.figure(figsize=(10, 10))
        for i in range(9):
            image = random_pics[i]
            image = tf.image.convert_image_dtype(image, tf.uint8)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image.numpy().astype("uint8"))
            plt.axis("off")
        plt.show()

    def on_train_batch_end(self, batch, logs={}):
        tf.py_function(func=self.print_pics, inp=[self.seen, self.display], Tout=[])
        self.seen += 1


class MemoryCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs={}):
        if batch % 100 == 0:
            print(' ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class Generator(torch.nn.Module):
    def __init__(self, batch_size, code_shape, noise_features, image_shape, decoder_filters_list, pixel_features,
                 eval_model):
        super(Generator, self).__init__()
        self.image_shape = image_shape
        self.pixel_features = pixel_features
        self.eval_model = eval_model
        dims = DimsTracker(input_dim=[batch_size, code_shape[1], code_shape[2],
                                      code_shape[3] + noise_features])
        units = code_shape[1] * code_shape[2] * (code_shape[3] + noise_features)
        self.linear1, dims = new_linear(dims, units)
        self.generator_blocks = torch.nn.ModuleList()
        current_decoder_layers = []
        for i, f in enumerate(decoder_filters_list):
            self.generator_blocks.append(GeneratorBlock(dims, f))
            current_decoder_layers.append(f)
            dims.update_dims(transform=lambda _: self.generator_blocks[-1].out_dims.curr_dim())
            print(f"Generator: dims={dims.curr_size()}, image shape={self.image_shape}")
            if dims.curr_size() == self.image_shape:
                break
        self.current_decoder_layers = current_decoder_layers
        self.pixel_features_conv, dims = new_conv2d(curr_dims=dims, filters=self.pixel_features, kernel_size=(1, 1))
        self.to_rgb, dims = new_conv2d(curr_dims=dims, filters=3, kernel_size=(1, 1))

        if eval_model:
            self.eval()

        self.out_dims = dims

    def forward(self, inputs):
        if self.eval_model:
            x = inputs
        else:
            x, alpha = inputs
        x = torch.reshape(x, shape=[inputs.shape[0], -1])
        x = self.linear1(x)
        x = torch.reshape(x, shape=inputs.shape)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        for block in self.generator_blocks:
            x_prev = x
            x = block(x)
        x = self.pixel_features_conv(x)
        x = torch.nn.ReLU()(x)
        x = self.to_rgb(x)
        if not self.eval_model:
            x_prev = torch.nn.UpsamplingNearest2d()(x_prev)
            x_prev = self.to_rgb(x_prev)
            x_prev = x_prev * (1 - alpha)
            x = x * alpha
            x = x + x_prev
        x = torch.nn.Tanh()(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, batch_size, image_shape, decoder_filters_list,
                 eval_model):
        super(Encoder, self).__init__()
        dims = DimsTracker(input_dim=[batch_size, image_shape[0], image_shape[1], 3])
        self.eval_model = eval_model

        current_encoder_layers = reversed(decoder_filters_list)
        f = next(current_encoder_layers)
        # average pooling reduces
        avg_pooling2d_dims = dims
        avg_pooling2d_dims.update_dims(DimsTransformer(size_transform=lambda x: int(x/2)))
        # fade in layers
        self.fromGRB, _ = new_conv2d(avg_pooling2d_dims, f, kernel_size=(1, 1))
        self.encoder_block = EncoderBlock(dims.curr_dim(), f)
        dims.update_dims(transform=lambda _: self.encoder_block.out_dims)
        self.encoder_blocks = []
        for i, f in enumerate(current_encoder_layers):
            self.encoder_blocks.append(EncoderBlock(dims.curr_dim(), f))
            dims.update_dims(transform=lambda _: self.encoder_blocks[-1].out_dims)
            print(f"Encoder: output shape={dims.curr_size()}")

        if eval_model:
            self.eval()

        self.output_dims = dims

    def forward(self, inputs):
        if not self.eval_model:
            x = inputs
        else:
            x, alpha = inputs
        x_prev = x
        x = self.encoder_block(x)
        if not self.eval_model:
            x_prev = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))(x_prev)
            x_prev = self.fromGRB(x_prev)
            x_prev = x_prev * (1 - alpha)
            x = x * alpha
            x = x + x_prev

        for block in self.encoder_blocks:
            x = block(x)
        return x


class CriticHead(torch.nn.Module):
    def __init__(self, batch_size, code_shape, current_decoder_layers=None):
        super(CriticHead, self).__init__()
        if current_decoder_layers is None:
            dims = DimsTracker(input_dim=[batch_size, code_shape[1], code_shape[2], code_shape[3]])
        else:
            dims = DimsTracker(input_dim=[batch_size, code_shape[1], code_shape[2],
                                          current_decoder_layers[0]])

        self.linear1, dims = new_linear(dims, units=1)
        self.output_dims = dims

    def forward(self, inputs):
        x = torch.reshape(inputs, shape=[inputs.shape[0], -1])
        x = self.linear1(x)
        return x


class EncoderHead(torch.nn.Module):
    def __init__(self, batch_size, code_shape, current_decoder_layers=None):
        super(EncoderHead, self).__init__()
        if current_decoder_layers is None:
            dims = DimsTracker(input_dim=[batch_size, code_shape[1], code_shape[2], code_shape[3]])
        else:
            dims = DimsTracker(input_dim=[batch_size, code_shape[1], code_shape[2],
                                          current_decoder_layers[0]])

        self.linear1, dims = new_linear(dims, units=code_shape[3])
        self.output_dims = dims

    def forward(self, inputs):
        x = torch.reshape(inputs, shape=[inputs.shape[0], -1])
        x = self.linear1(x)
        return x


class FacesDataset(Dataset):
    def __init__(self, files, size=(128, 128)):
        self.files = files
        self.ln = len(files)
        self.size = size

    def update_size(self, size):
        self.size = size

    def __getitem__(self, item):
        file = self.files[item]
        file = torchvision.io.read_file(file)
        image = torchvision.io.decode_jpeg(file)
        image = torchvision.transforms.Resize(size=self.size)(image)
        return torchvision.transforms.ConvertImageDtype(torch.float32)(image)

    def __len__(self):
        return self.ln


class InfoWGAN:
    def __init__(self, files_dir, batch_size, code_features, noise_features, pixel_features,
                 decoder_filters_list, cp_dir, epochs_per_phase=5, training_file=None):
        super(InfoWGAN, self).__init__()
        self.code_shape = [None, 1, 1, code_features]
        self.noise_features = noise_features
        self.code_features = code_features
        self.pixel_features = pixel_features
        self.batch_size = batch_size

        self.cp_dir = cp_dir
        self.training_file = training_file
        self.epochs_per_phase = epochs_per_phase
        self.curr_epoch = 0

        self.decoder_filters_list = decoder_filters_list

        self.real_label = torch.Tensor(-1, dtype=torch.float32)
        self.fake_label = torch.Tensor(1, dtype=torch.float32)

        self.current_alpha = None

        self.d_optimizer = torch.optim.Adam(params={})
        self.g_optimizer = torch.optim.Adam(params={})
        self.q_optimizer = torch.optim.Adam(params={})
        self.gradLAMBDA = torch.Tensor(0.1)
        self.infoLAMBDA = torch.Tensor(0.1)
        self.generator = torch.nn.Module()
        self.coder = torch.nn.Module()
        self.coderHead = torch.nn.Module()
        self.criticHead = torch.nn.Module()

        self.image_shape = None

        self.files_dir = files_dir

    def build_models(self, image_shape, eval_model):
        self.generator = Generator(self.batch_size, code_shape=self.code_shape,
                                   noise_features=self.noise_features, image_shape=image_shape,
                                   decoder_filters_list=self.decoder_filters_list,
                                   pixel_features=self.pixel_features,
                                   eval_model=eval_model)
        current_decoder_layers = self.generator.current_decoder_layers
        self.coder = Encoder(batch_size=self.batch_size, image_shape=image_shape,
                             decoder_filters_list=current_decoder_layers, eval_model=eval_model)
        self.coderHead = EncoderHead(batch_size=self.batch_size, code_shape=self.code_shape,
                                     current_decoder_layers=current_decoder_layers)
        self.criticHead = CriticHead(batch_size=self.batch_size, code_shape=self.code_shape,
                                     current_decoder_layers=current_decoder_layers)

    @staticmethod
    def get_image_dataset(files_dir):
        files = glob.glob(files_dir)
        image_count = len(files)
        # train/test split
        train_perc = 0.8
        train_samples = int(round(image_count * train_perc))
        train_files = files[:train_samples]
        val_files = files[train_samples:]
        train_dataset = FacesDataset(train_files)
        val_dataset = FacesDataset(val_files)
        return train_dataset, val_dataset

    def fit(self, files_dir):
        '''
        Custom training loop.
        wrap the Model.fit function to first call build_models on the correct image shape and
        load models as needed.
        Adjust image inputs to correct resolution
        '''

        # 1 image batch at 128x128... 256 images at 4x4
        images_per_batch = int(max(1,
                                   int(0.25 * 128 * 128 / (self.image_shape[0] * self.image_shape[1]))))
        while True:
            # build models
            self.load_image_shape()
            train_ds, eval_ds = self.get_image_dataset(self.files_dir)
            self.build_models(self.image_shape, eval_model=False)
            self.compile(d_optimizer=torch.optim.Adam(chain(self.criticHead.parameters(),
                                                            self.coder.parameters()),
                                                      lr=0.001),
                         g_optimizer=torch.optim.Adam(self.generator.parameters(), lr=0.001),
                         q_optimizer=torch.optim.Adam(self.coderHead.parameters(), lr=0.001),
                         grad_lambda=0.001, info_lambda=0.001)
            # load checkpoint
            start_epoch = self.load_cp()

            # get datasets of resized images
            train_loader = DataLoader(train_ds, batch_size=images_per_batch, shuffle=True)
            losses = None
            for epoch in range(start_epoch, self.epochs_per_phase):
                self.current_alpha = (self.curr_epoch % self.epochs_per_phase + 1) / self.epochs_per_phase
                train_ds = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
                for i, batch in enumerate(train_loader):
                    losses = self.train_step(batch)

            # checkpoint model
            self.save_cp(0, losses)

            # update image shape
            image_shape_prev = self.image_shape
            self.image_shape = [image_shape_prev[0] * 2, image_shape_prev[0] * 2]
            if self.image_shape[0] >= 128:
                print("Reached max image resolution. Done training")
                break
            train_ds.update_size(self.image_shape)

    def save_cp(self, epoch, losses):
        cp_dict = {}
        cp_dict += losses
        if not os.path.exists(self.cp_dir):
            os.mkdir(self.cp_dir)
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'coder_state_dict': self.coder.state_dict(),
            'critic_head_state_dict': self.criticHead.state_dict(),
            'coder_head_state_dict': self.coderHead.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'image_shape': self.image_shape
        }, os.path.join(self.cp_dir, "checkpoint.pth"))

    def load_cp(self, load_optimizer_state=False):
        if os.path.exists(os.path.join(self.cp_dir, "checkpoint.pth")):
            checkpoint = torch.load(os.path.join(self.cp_dir, "checkpoint.pth"))
            if checkpoint['epoch'] == 0:
                # purge fade in layers from checkpoint dict so they are not loaded
                for outer_key in checkpoint.keys():
                    for key in checkpoint[outer_key].keys():
                        if "rgb" in key:
                            checkpoint[outer_key].pop(key)
                        if "pixel_features_conv" in key:
                            checkpoint[outer_key].pop(key)

            self.generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
            self.coder.load_state_dict(checkpoint['coder_state_dict'], strict=False)
            self.criticHead.load_state_dict(checkpoint['critic_head_state_dict'], strict=False)
            self.coderHead.load_state_dict(checkpoint['coder_head_state_dict'], strict=False)
            self.image_shape = checkpoint['image_shape']
            if load_optimizer_state:
                self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
            return checkpoint['epoch']
        else:
            print("checkpoint file not found. Starting training from scratch")
            return 0  # start epoch = 0

    def load_image_shape(self):
        if os.path.exists(os.path.join(self.cp_dir, "checkpoint.pth")):
            checkpoint = torch.load(os.path.join(self.cp_dir, "checkpoint.pth"))
            self.image_shape = checkpoint['image_shape']
            print(f"Found checkpoint. Starting image shape={self.image_shape}")
        else:
            self.image_shape = [4, 4]
            print(f"Checkpoint file not found. Starting image shape={self.image_shape}")

    def compile(self, d_optimizer: torch.optim.Optimizer,
                g_optimizer: torch.optim.Optimizer,
                q_optimizer: torch.optim.Optimizer,
                grad_lambda: float,
                info_lambda: float):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.q_optimizer = q_optimizer
        self.gradLAMBDA = torch.Tensor(grad_lambda, dtype=torch.float32)
        self.infoLAMBDA = torch.Tensor(info_lambda, dtype=torch.float32)

    def test_step(self, data):
        if isinstance(data, tuple):
            real_images = data[0]
        else:
            real_images = data

        # progressive GAN alpha hyperparameter
        with torch.no_grad():
            current_epoch = torch.Tensor(self.curr_epoch, dtype=torch.float32)
            alpha = torch.divide(torch.add(torch.fmod(current_epoch, self.epochs_per_phase), 1), self.epochs_per_phase)
            # Sample random points in the latent space
            batch_size = tf.shape(real_images)[0]
            latent_shape = [batch_size, self.code_shape[1], self.code_shape[2],
                            self.code_features + self.noise_features]
            random_latent_vectors = torch.randn(size=latent_shape)

            # Decode them to fake images
            generated_images = self.generator([random_latent_vectors, alpha])

            # generate random "intermidiate" images interpolating the generated and real images for gradient penalty
            eps = torch.rand([batch_size, 1, 1, 1], dtype=torch.float32)
            interp_images = torch.mul(eps, real_images) + torch.mul((1 - eps), generated_images)

            # Combine them with real images
            combined_images = torch.cat([generated_images, real_images], dim=0)

            # Assemble labels discriminating real from fake images
            labels = torch.cat([self.fake_label * torch.ones((batch_size, 1), dtype=torch.float32),
                                self.real_label * torch.ones((batch_size, 1), dtype=torch.float32)], dim=0)

            conv_out = self.coder([combined_images, alpha * tf.ones(2 * batch_size)])
            criticism = self.criticHead(conv_out)
            wgan_loss = torch.mean(labels * criticism)
        # get grad_x(critic(interpolated_images))
        interp_images.requires_grad(True)
        for param in self.coder.parameters():
            param.requires_grad(False)
        for param in self.criticHead.parameters():
            param.requires_grad(False)
        interp_conv = self.coder([interp_images, alpha])
        interp_criticism = self.criticHead(interp_conv)
        # d_crit/d_img
        for elem in interp_criticism:
            elem.backward()
        critic_x_grad = interp_images.grad
        interp_images.grad.zero_()
        for param in self.coder.parameters():
            param.requires_grad(True)
        for param in self.criticHead.parameters():
            param.requires_grad(True)

        with torch.no_grad():
            critic_x_grad = torch.reshape(critic_x_grad, [batch_size, -1])
            penalty_loss = torch.mean(torch.square(torch.add(torch.norm(critic_x_grad, dim=-1, keepdim=True), -1)))
            d_loss = wgan_loss + self.gradLAMBDA * penalty_loss

            # Sample random points in the latent space
            random_latent_vectors = torch.randn(size=latent_shape)
            random_code = random_latent_vectors[:, :, :, :self.code_features]

            # Assemble labels that say "all real images"
            # This makes the generator want to create real images (match the label) since
            # we do not include an additional minus in the loss
            misleading_labels = self.real_label * torch.ones([batch_size, 1, 1, 1])

            # Train the generator and encoder(note that we should *not* update the weights
            # of the critic or encoder)!
            fake_images = self.generator([random_latent_vectors, alpha])
            conv = self.coder([fake_images, alpha])
            criticism = self.criticHead(conv)
            code_prediction = self.coderHead(conv)
            g_loss = torch.mean(misleading_labels * criticism)
            info_loss = tf.reduce_mean(tf.math.squared_difference(code_prediction, random_code))

        return {"critic_loss": -wgan_loss, "generator_loss": g_loss,
                #                 "info_loss": info_loss,
                "gradient_penalty_loss": penalty_loss}

    def train_step(self, image):
        curr_epoch = torch.Tensor(self.curr_epoch, dtype=torch.int32)
        # progressive GAN alpha hyperparameter
        alpha = torch.divide(torch.add(torch.fmod(curr_epoch, self.epochs_per_phase), 1), self.epochs_per_phase)
        # Sample random points in the latent space
        batch_size = image.shape
        latent_shape = (batch_size, self.code_shape[1], self.code_shape[2],
                        self.code_features + self.noise_features)
        random_latent_vectors = torch.randn(size=latent_shape)

        # Decode them to fake images
        generated_images = self.generator([random_latent_vectors, alpha])

        # generate random "intermidiate" images interpolating the generated and real images for gradient penalty
        eps = torch.rand(size=[batch_size, 1, 1, 1])
        interp_images = torch.mul(eps, image) + torch.mul((1 - eps), generated_images)

        # Combine them with real images
        combined_images = torch.cat([generated_images, image], dim=0)

        # Assemble labels discriminating real from fake images
        labels = torch.cat([self.fake_label * torch.ones([batch_size, 1]),
                            self.real_label * torch.ones([batch_size, 1])], dim=0)

        # Train the discriminator to optimality
        wgan_loss = torch.Tensor(0, dtype=torch.float32)
        penalty_loss = torch.Tensor(0, dtype=torch.float32)
        for step in range(5):
            conv_out = self.coder([combined_images, alpha])
            criticism = self.criticHead(conv_out)
            wgan_loss = torch.mean(torch.mul(labels, criticism))

            # get grad_x(critic(interpolated_images))
            interp_images.requires_grad(True)
            conv_out_interp = self.coder([interp_images, alpha])
            interp_criticism = self.criticHead(conv_out_interp)
            for elem in interp_criticism:
                elem.backward()
            critic_x_grad = interp_images.grad
            interp_images.grad.zero_()
            critic_x_grad = torch.reshape(critic_x_grad, [batch_size, -1])
            penalty_loss = torch.mean(torch.square(torch.add(torch.norm(critic_x_grad, dim=-1, keepdim=True), -1)))

            d_loss = wgan_loss + self.gradLAMBDA * penalty_loss
            d_loss.backward()
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()

        # Sample random points in the latent space
        random_latent_vectors = torch.randn(size=latent_shape)
        random_code = random_latent_vectors[:, :, :, :self.code_features]

        # Assemble labels that say "all real images"
        # This makes the generator want to create real images (match the label) since
        # we do not include an additional minus in the loss
        misleading_labels = self.real_label * torch.ones([batch_size, 1])

        # Train the generator and encoder(note that we should *not* update the weights
        # of the critic or encoder)!
        fake_images = self.generator([random_latent_vectors, alpha])
        conv_out_fake = self.coder([fake_images, alpha])
        fake_criticism = self.criticHead(conv_out_fake)
        code_pred = self.coderHead(conv_out_fake)
        g_loss = torch.mean(misleading_labels * fake_criticism)
        info_loss = torch.mean(torch.square(torch.diff(code_pred, random_code)))
        total_g_loss = g_loss + self.infoLAMBDA * info_loss
        total_g_loss.backward()
        self.g_optimizer.step()
        self.g_optimizer.zero_grad()
        self.q_optimizer.step()
        self.q_optimizer.zero_grad()

        return {"critic_loss": -wgan_loss, "generator_loss": g_loss,
                #                 "info_loss": info_loss,
                "gradient_penalty_loss": penalty_loss}
