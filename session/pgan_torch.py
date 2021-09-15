import os
import copy
import glob
from typing import Iterable
from torch._C import device, dtype
from torch.autograd import grad
from tqdm import tqdm
from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import functools
import collections
import torch

C_AXIS = 1


class DimsTracker:
    def __init__(self, input_dim: list, c_axis=C_AXIS):
        self.curr_dim = input_dim
        self.prev_dim = input_dim
        self.c_axis = c_axis

    def update_dims(self, transform):
        self.prev_dim = self.curr_dim
        self.curr_dim = transform(self)

    def get_dims(self):
        return self.prev_dim, self.curr_dim

    def prev_channels(self):
        return self.prev_dim[self.c_axis]

    def curr_channels(self):
        return self.curr_dim[self.c_axis]

    def curr_size(self):
        return [dim for i, dim in enumerate(self.curr_dim) if i not in [0, self.c_axis]]

    def prev_size(self):
        return [dim for i, dim in enumerate(self.prev_dim) if i not in [0, self.c_axis]]

    def num_prev_features(self):
        return functools.reduce(lambda x, y: x+y, self.prev_dim)

    def num_curr_features(self):
        return functools.reduce(lambda x, y: x*y, self.curr_dim[1:])


class DimsTransformer:
    def __init__(self, batch_transform=None, size_transform=None, channels_transform=None, c_axis=C_AXIS):
        self.c_axis = c_axis
        self.size_transform = size_transform
        self.channels_transform = channels_transform
        self.batch_transform = batch_transform

    def __call__(self, dims_tracker: DimsTracker):
        dims = dims_tracker.curr_dim
        channels = dims[self.c_axis] if self.channels_transform is None else self.channels_transform(
            dims[self.c_axis])
        batch_size = dims[0] if self.batch_transform is None else self.batch_transform(
            dims[0])
        other_axes = (ax for ax in range(len(dims))
                      if ax not in [0, self.c_axis])
        other_dims = [dims[ax] for ax in other_axes]
        size = other_dims if self.size_transform is None else self.size_transform(
            other_dims)
        if self.c_axis == 1:
            new_size = [batch_size, channels, *size]
        else:
            new_size = [batch_size, *size, channels]
        return new_size


class FlattenTransformer(DimsTransformer):
    def __init__(self, c_axis=C_AXIS):
        super(FlattenTransformer, self).__init__(batch_transform=None, size_transform=None, channels_transform=None,
                                                 c_axis=c_axis)

    def __call__(self, dims_tracker: DimsTracker):
        dims = dims_tracker.curr_dim
        batch_size = dims[0]
        other_axes = (ax for ax in range(len(dims))
                      if ax not in [0, self.c_axis])
        size = [dims[ax] for ax in other_axes]
        size_mul = functools.reduce(lambda x, y: x * y, size)
        features = size_mul * dims[self.c_axis]
        new_size = [batch_size, features]
        return new_size


class Conv2dTransformer(DimsTransformer):
    def __init__(self, filters, kernel_size, stride, padding="same", dilation=(1, 1)):
        def size_transform(input_size, pad=padding):
            if pad == "same":
                if stride != (1, 1):
                    raise ValueError(
                        "conv cannot have same padding with stride > 1")
                return input_size
            elif pad == "valid":
                pad = (0, 0)

            def output_size(in_size: int, ax: int, p=pad):
                out_size = in_size + 2 * \
                    p[ax] - dilation[ax] * (kernel_size[ax] - 1) - 1
                out_size = out_size / stride[ax]
                out_size += 1
                out_size = int(out_size)
                return out_size

            return [output_size(ax, i) for i, ax in enumerate(input_size)]

        def channels_transform(input_filters):
            return filters

        super(Conv2dTransformer, self).__init__(size_transform=size_transform, channels_transform=channels_transform,
                                                c_axis=C_AXIS)


class CatTransformer(DimsTransformer):
    def __init__(self, dims_list: list, cat_dim):
        new_dim = functools.reduce(
            lambda x, y: x.curr_dim[cat_dim] + y.curr_dim[cat_dim], dims_list)
        rest_dim = [dims_list[0].curr_dim[ax]
                    for ax in range(len(dims_list[0].curr_dim)) if ax != cat_dim]
        self.out_dim = rest_dim
        self.out_dim.insert(cat_dim, new_dim)
        super(CatTransformer, self).__init__()

    def __call__(self, dims_tracker: DimsTracker):
        return self.out_dim


class ConvTrans2dTransformer(DimsTransformer):
    def __init__(self, filters, kernel_size, stride,
                 padding=(0, 0), dilation=(1, 1), output_padding=(0, 0)):
        def size_transform(input_size, pad=padding):
            if pad == "same":
                if stride != (1, 1):
                    raise ValueError(
                        "conv cannot have same padding with stride > 1")
                return input_size
            elif pad == "valid":
                pad = (0, 0)

            def output_size(in_size: int, ax: int, p=pad):
                out_size = (in_size - 1) * stride[ax] - 2 * p[ax] + \
                    dilation[ax] * (kernel_size[ax] - 1) + \
                    output_padding[ax] + 1
                return out_size

            return [output_size(ax, i) for i, ax in enumerate(input_size)]

        def channels_transform(input_filters):
            return filters

        super(ConvTrans2dTransformer, self).__init__(size_transform=size_transform,
                                                     channels_transform=channels_transform, c_axis=C_AXIS)


def new_conv2d(curr_dims: DimsTracker, filters, kernel_size=(3, 3), stride=(1, 1), padding="same", dilation=(1, 1)):
    transform = Conv2dTransformer(
        filters, kernel_size=kernel_size, padding=padding, stride=stride)
    conv = torch.nn.Conv2d(curr_dims.curr_channels(), filters, kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation)
    curr_dims.update_dims(transform)
    return conv, curr_dims


def new_conv_trans2d(curr_dims: DimsTracker, filters, kernel_size=(3, 3), stride=(2, 2),
                     padding=(1, 1), dilation=(1, 1), output_padding=(1, 1)):
    transform = ConvTrans2dTransformer(filters, kernel_size=kernel_size, padding=padding,
                                       stride=stride, output_padding=output_padding, dilation=dilation)
    conv = torch.nn.ConvTranspose2d(curr_dims.curr_channels(), filters, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, output_padding=output_padding)
    curr_dims.update_dims(transform)
    return conv, curr_dims


def new_linear(curr_dims: DimsTracker, units):
    linear = torch.nn.Linear(
        in_features=curr_dims.num_curr_features(), out_features=units)
    curr_dims = DimsTracker([curr_dims.curr_dim[0], units])
    return linear, curr_dims


class EncoderBlock(torch.nn.Module):
    def __init__(self, input_dim: list, filters, c_axis=C_AXIS):
        super(EncoderBlock, self).__init__()
        dims = DimsTracker(input_dim)
        input_dims = copy.copy(dims)
        self.conv1, dims = new_conv2d(dims, filters=filters)
        self.ln1 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim[1:])
        self.conv2, dims = new_conv2d(dims, filters=filters)
        self.ln2 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim[1:])
        # cat here
        dims.update_dims(transform=CatTransformer(
            [dims, input_dims], cat_dim=c_axis))
        self.conv3, dims = new_conv2d(dims, filters=filters, stride=(
            2, 2), padding=(1, 1))  # output image size halved
        self.ln3 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim[1:])

        self.out_dims = dims

    def forward(self, inputs):
        inputs_cp = torch.clone(inputs)
        x = self.conv1(inputs)  # output_dim_pre_concat
        x = torch.nn.LeakyReLU(negative_slope=0.1)(x)
        x = self.ln1(x)
        x = self.conv2(x)
        x = torch.nn.LeakyReLU(negative_slope=0.1)(x)
        x = self.ln2(x)
        # skip connection, C_AXIS: filters * 2
        x = torch.cat([x, inputs_cp], dim=C_AXIS)
        x = self.conv3(x)
        x = torch.nn.LeakyReLU(negative_slope=0.1)(x)
        x = self.ln3(x)
        return x


class GeneratorBlock(torch.nn.Module):
    def __init__(self, input_dim: list, filters):
        super(GeneratorBlock, self).__init__()
        dims = DimsTracker(input_dim)
        self.conv1, dims = new_conv_trans2d(
            dims, filters=filters)  # doubles input size
        self.ln1 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim[1:])
        self.conv2, dims = new_conv2d(curr_dims=dims, filters=filters)
        self.ln2 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim[1:])
        self.conv3, dims = new_conv2d(curr_dims=dims, filters=filters)
        self.ln3 = torch.nn.LayerNorm(normalized_shape=dims.curr_dim[1:])

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


class Generator(torch.nn.Module):
    def __init__(self, batch_size, code_shape, noise_features, image_shape, decoder_filters_list, pixel_features,
                 eval_model):
        super(Generator, self).__init__()
        self.image_shape = image_shape
        self.pixel_features = pixel_features
        self.eval_model = eval_model
        size = [code_shape[1], code_shape[2]]
        channels = code_shape[3] + noise_features
        if C_AXIS == 1:
            inp_dims = DimsTracker(input_dim=[batch_size, channels, *size])
        else:
            inp_dims = DimsTracker(input_dim=[batch_size, *size, channels])
        print("building generator")
        print(f"input shape {inp_dims.curr_dim}")
        print("-------------------")
        print(f"layer           out_shape")
        dims = copy.copy(inp_dims)
        dims.update_dims(FlattenTransformer())
        print(f"flatten           {dims.curr_dim}")
        units = dims.num_curr_features()
        self.linear1, dims = new_linear(dims, units)
        print(f"linear           {dims.curr_dim}")
        self.generator_blocks = torch.nn.ModuleList()
        # reshape back to image shape
        dims = copy.copy(inp_dims)
        print(f"reshape           {dims.curr_dim}")
        current_decoder_inputs = []
        decoder_filters_iter = iter(decoder_filters_list)
        for i, f in enumerate(decoder_filters_iter):
            self.generator_blocks.append(GeneratorBlock(dims.curr_dim, f))
            dims_prev = copy.copy(dims)
            dims.update_dims(
                transform=lambda _: self.generator_blocks[-1].out_dims.curr_dim)
            print(f"block_{i}           {dims.curr_dim}")
            current_decoder_inputs.append(dims.prev_channels())
            if tuple(dims.curr_size()) == self.image_shape:
                break

        # output channels of first encoder block
        self.from_rgb_channels = dims.curr_channels()
        self.current_decoder_inputs = current_decoder_inputs

        self.pixel_features_conv, dims = new_conv2d(
            curr_dims=dims, filters=self.pixel_features, kernel_size=(1, 1))
        print(f"pixel_features_conv           {dims.curr_dim}")
        self.to_rgb, dims = new_conv2d(
            curr_dims=dims, filters=3, kernel_size=(1, 1))
        print(f"to_rgb           {dims.curr_dim}")
        # upsample x_prev
        dims_prev.update_dims(transform=DimsTransformer(
            size_transform=lambda xs: [x * 2 for x in xs]))
        print(f"upsample2d           {dims_prev.curr_dim}")
        self.to_rgb_prev, to_rgb_prev_dims = new_conv2d(
            curr_dims=dims_prev, filters=3, kernel_size=(1, 1))
        print(f"to_rgb_prev           {to_rgb_prev_dims.curr_dim}")
        print("\n")
        if eval_model:
            self.eval()

        self.out_dims = dims

    def forward(self, inputs: torch.Tensor):
        if self.eval_model:
            x_inp = inputs
        else:
            x_inp, alpha = inputs
        x = torch.reshape(x_inp, shape=[x_inp.shape[0], -1])
        x = self.linear1(x)
        x = torch.reshape(x, shape=x_inp.shape)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        for block in self.generator_blocks:
            x_prev = torch.clone(x)
            x = block(x)
        x = self.pixel_features_conv(x)
        x = torch.nn.ReLU()(x)
        x = self.to_rgb(x)
        if not self.eval_model:
            x_prev = torch.nn.UpsamplingNearest2d(scale_factor=2)(x_prev)
            x_prev = self.to_rgb_prev(x_prev)
            x_prev = x_prev * (1 - alpha)
            x = x * alpha
            x = x + x_prev
        x = torch.nn.Tanh()(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, batch_size, image_shape, current_decoder_inputs, from_rgb_channels,
                 eval_model):
        super(Encoder, self).__init__()
        size = [image_shape[0], image_shape[1]]
        channels = 3
        if C_AXIS == 1:
            dims = DimsTracker(input_dim=[batch_size, channels, *size])
        else:
            dims = DimsTracker(input_dim=[batch_size, *size, channels])
        self.eval_model = eval_model
        print("building encoder")
        print(f"input shape {dims.curr_dim}")
        print("-------------------")
        print(f"layer           out_shape           parameters shape")

        current_encoder_layers = reversed(current_decoder_inputs)
        f = next(current_encoder_layers)
        # fade in layers
        dims_prev = copy.copy(dims)
        self.from_rgb, dims = new_conv2d(
            dims, from_rgb_channels, kernel_size=(1, 1))
        print(
            f"from_rgb           {dims.curr_dim}         {next(self.from_rgb.parameters()).shape}")
        self.from_rgb_prev, dims_prev = new_conv2d(
            dims_prev, f, kernel_size=(1, 1))
        print(
            f"from_rgb_prev      {dims_prev.curr_dim}         {next(self.from_rgb_prev.parameters()).shape}")
        # average pooling halves size
        dims_prev.update_dims(
            DimsTransformer(size_transform=lambda x: [int(x[el] // 2) for el in range(len(x))]))
        avg_pooling2d_dims = dims_prev
        print(f"average pooling           {avg_pooling2d_dims.curr_dim}")
        encoder_blocks = [EncoderBlock(dims.curr_dim, f)]
        dims.update_dims(
            transform=lambda _: encoder_blocks[-1].out_dims.curr_dim)

        print(f"block_0       {dims.curr_dim}")
        for i, f in enumerate(current_encoder_layers):
            encoder_blocks.append(EncoderBlock(dims.curr_dim, f))
            dims.update_dims(
                transform=lambda _: encoder_blocks[-1].out_dims.curr_dim)
            print(f"block_{i+1}       {dims.curr_dim}")
        # We save the blocks in reverse order, so that when we save the oldest block will have index zero.
        # This way the existing blocks don't change when we add additional ones
        self.encoder_blocks_reverse = torch.nn.ModuleList(
            reversed(encoder_blocks))
        print("\n")
        if eval_model:
            self.eval()

        self.output_dims = dims

    def forward(self, inputs):
        if self.eval_model:
            x = inputs
        else:
            x, alpha = inputs
        x_prev = torch.clone(x)
        x = self.from_rgb(x)
        x = self.encoder_blocks_reverse[-1](x)
        if not self.eval_model:
            x_prev = self.from_rgb_prev(x_prev)
            x_prev = torch.nn.AvgPool2d(kernel_size=(
                2, 2), stride=(2, 2), padding=(0, 0))(x_prev)
            x_prev = x_prev * (1 - alpha)
            x = x * alpha
            x = x + x_prev
        for i in reversed(range(len(self.encoder_blocks_reverse)-1)):
            x = self.encoder_blocks_reverse[i](x)
        return x


class CriticHead(torch.nn.Module):
    def __init__(self, batch_size, code_shape, current_decoder_layers=None):
        super(CriticHead, self).__init__()
        if current_decoder_layers is None:
            dims = DimsTracker(
                input_dim=[batch_size, code_shape[1], code_shape[2], code_shape[3]])
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
            dims = DimsTracker(
                input_dim=[batch_size, code_shape[1], code_shape[2], code_shape[3]])
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
    def __init__(self, files_dir, code_features, noise_features, pixel_features,
                 decoder_filters_list, cp_dir, epochs_per_phase=5, batch_sizes=None,
                 info_lambda=100,
                 grad_lambda=10,
                 lr=0.001,
                 adaptive_gradient_clipping=False,
                 gradient_centralization=False,
                 start_image_shape=(4, 4),
                 max_image_shape=128,
                 start_from_next_resolution=False):
        super(InfoWGAN, self).__init__()
        self.max_image_shape = max_image_shape
        self.start_image_shape = start_image_shape
        self.start_from_next_resolution = start_from_next_resolution
        self.image_shape = start_image_shape
        self.code_shape = [None, 1, 1, code_features]
        self.noise_features = noise_features
        self.code_features = code_features
        self.pixel_features = pixel_features
        self.batch_sizes = batch_sizes
        if self.batch_sizes is None:
            self.batch_sizes = {(4, 4): 650,
                                (8, 8): 650,
                                (16, 16): 512,
                                (32, 32): 512,
                                (64, 64): 512,
                                (128, 128): 512}

        self.cp_dir = cp_dir
        if not os.path.exists(cp_dir):
            os.mkdir(cp_dir)

        self.tensorboard_writer = None

        self.epochs_per_phase = epochs_per_phase
        self.curr_epoch = 0

        self.decoder_filters_list = decoder_filters_list

        self.real_label = torch.tensor(-1, dtype=torch.float32)
        self.fake_label = torch.tensor(1, dtype=torch.float32)

        self.current_alpha = None

        self.d_optimizer = None
        self.g_optimizer = None
        self.q_optimizer = None
        self.grad_lambda = torch.tensor(info_lambda)
        self.info_lambda = torch.tensor(grad_lambda)
        self.generator = torch.nn.Module()
        self.coder = torch.nn.Module()
        self.coderHead = torch.nn.Module()
        self.criticHead = torch.nn.Module()

        self.lr = lr
        self.adaptive_gradient_clipping = adaptive_gradient_clipping
        self.gradient_centralization = gradient_centralization

        self.files_dir = files_dir

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_models(self, image_shape, eval_model=False):
        """
        initialize models and send to device
        :param image_shape: current image shape in pGAN training
        :param eval_model: return evaluation models if True. defaults to False
        :return: None
        """
        self.generator = Generator(self.batch_sizes[image_shape], code_shape=self.code_shape,
                                   noise_features=self.noise_features, image_shape=image_shape,
                                   decoder_filters_list=self.decoder_filters_list,
                                   pixel_features=self.pixel_features,
                                   eval_model=eval_model)
        current_decoder_inputs = self.generator.current_decoder_inputs
        from_rgb_channels = self.generator.from_rgb_channels
        self.coder = Encoder(batch_size=self.batch_sizes[image_shape], image_shape=image_shape,
                             current_decoder_inputs=current_decoder_inputs, from_rgb_channels=from_rgb_channels,
                             eval_model=eval_model)
        self.coderHead = EncoderHead(batch_size=self.batch_sizes[image_shape], code_shape=self.code_shape,
                                     current_decoder_layers=current_decoder_inputs)
        self.criticHead = CriticHead(batch_size=self.batch_sizes[image_shape], code_shape=self.code_shape,
                                     current_decoder_layers=current_decoder_inputs)

    def models_to_device(self):
        self.generator.to(self.device)
        self.coder.to(self.device)
        self.coderHead.to(self.device)
        self.criticHead.to(self.device)

    def get_image_dataset(self, files_dir):
        files = glob.glob(files_dir)
        image_count = len(files)
        # train/test split
        train_percent = 0.8
        train_samples = int(round(image_count * train_percent))
        train_files = files[:train_samples]
        val_files = files[train_samples:]
        train_dataset = FacesDataset(train_files, size=self.image_shape)
        val_dataset = FacesDataset(val_files, size=self.image_shape)
        return train_dataset, val_dataset

    def fit(self):
        """
        Custom training loop.
        load models as needed.
        Adjust image inputs to correct resolution
        :return:
        """
        test_steps = 5
        while True:
            # build models
            finished = self.load_image_shape()
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.cp_dir, f"summaries_{self.image_shape[0]}_{self.image_shape[1]}"))
            if finished:
                break
            print(f"learning rate = {self.lr}")
            train_ds, eval_ds = self.get_image_dataset(self.files_dir)
            self.build_models(self.image_shape, eval_model=False)
            self.compile(d_optimizer=torch.optim.Adam(chain(self.criticHead.parameters(),
                                                            self.coder.parameters()),
                                                      lr=self.lr),
                         g_optimizer=torch.optim.Adam(
                             self.generator.parameters(), lr=self.lr),
                         q_optimizer=torch.optim.Adam(
                             self.coderHead.parameters(), lr=self.lr),
                         grad_lambda=0.001, info_lambda=0.001)
            # load checkpoint
            start_epoch = self.load_cp()
            if self.start_from_next_resolution:
                self.start_from_next_resolution = False  # only do once
            self.models_to_device()

            losses = None
            running = {}
            for epoch in range(start_epoch, self.epochs_per_phase):
                self.current_alpha = (
                    self.curr_epoch % self.epochs_per_phase + 1) / self.epochs_per_phase
                # get datasets of resized images
                train_loader = DataLoader(
                    train_ds, batch_size=self.batch_sizes[self.image_shape], shuffle=True, num_workers=3)
                n_total_steps = len(train_loader)
                progress_bar = tqdm(train_loader)
                description = f"Epoch {epoch}"
                progress_bar.set_description(description)
                i = 0
                for batch in progress_bar:
                    batch = batch.to(self.device)
                    losses, metrics = self.train_step(batch)

                    global_step = epoch * n_total_steps + i
                    for scalar, value in chain(iter(losses.items()), iter(metrics.items())):
                        if scalar not in running.keys():
                            running[scalar] = 0.0
                        running[scalar] += value.item()

                    if (i + 1) % (n_total_steps // 4) == 0:
                        for scalar, value in running.items():
                            self.tensorboard_writer.add_scalars(scalar,
                                                                {"train": value / (n_total_steps // 4)}, global_step)
                        self.write_tensorboard_summaries(
                            batch[:9], global_step)
                        for scalar in running:
                            running[scalar] = 0.0
                    # save once mid epoch
                    if (i + 1) % (n_total_steps // 2.1) == 0 and (i + 1) // (n_total_steps // 2.1) == 1:
                        # test model and print scalars
                        eval_running = {}
                        eval_loader = DataLoader(
                            eval_ds, batch_size=self.batch_sizes[self.image_shape], shuffle=True)
                        it = iter(eval_loader)
                        for step in range(test_steps):
                            images = next(it)
                            images = images.to(self.device)
                            eval_losses, eval_metrics = self.test_step(
                                images)
                            for scalar, value in chain(iter(eval_losses.items()), iter(eval_metrics.items())):
                                if scalar not in eval_running.keys():
                                    eval_running[scalar] = 0.0
                                eval_running[scalar] += value.item()
                        for scalar, value in eval_running.items():
                            # description += f"{scalar} {value / test_steps:.3f}; "
                            self.tensorboard_writer.add_scalars(scalar,
                                                                {"test": value / test_steps}, global_step)
                        for scalar in running:
                            eval_running[scalar] = 0.0
                        # checkpoint model
                        self.save_cp(epoch, losses)
                    i += 1
                progress_bar.close()

            # checkpoint model
            self.save_cp('end', losses)

        self.tensorboard_writer.close()

    def save_cp(self, epoch, losses):
        cp_dict = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'coder_state_dict': self.coder.state_dict(),
            'critic_head_state_dict': self.criticHead.state_dict(),
            'coder_head_state_dict': self.coderHead.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'image_shape': self.image_shape
        }
        cp_dict.update(losses)
        if not os.path.exists(self.cp_dir):
            os.mkdir(self.cp_dir)
        torch.save(cp_dict, os.path.join(
            self.cp_dir, f"checkpoint_{self.image_shape[0]}_{self.image_shape[1]}.pth"))

    def get_latest_checkpoint(self):
        # default value
        curr_checkpoint = f"checkpoint_{self.start_image_shape[0]}_{self.start_image_shape[1]}.pth"
        if os.path.exists(self.cp_dir):
            # find latest checkpoint

            checkpoints = glob.glob(os.path.join(
                self.cp_dir, "checkpoint_*_*.pth"))
            if len(checkpoints) > 0:
                def get_image_shape_from_cp(cp):
                    fn = cp.split(sep='/')[-1]
                    fn = fn.split(sep='.')[0]
                    _, i, _ = fn.split(sep='_')
                    return int(i)
                checkpoints_by_image_shape = {
                    get_image_shape_from_cp(cp): cp for cp in checkpoints}
                max_image_shape = max(checkpoints_by_image_shape.keys())
                curr_checkpoint = checkpoints_by_image_shape[max_image_shape]

        return curr_checkpoint

    def load_cp(self, load_optimizer_state=False):
        curr_checkpoint = self.get_latest_checkpoint()
        if os.path.exists(os.path.join(self.cp_dir, curr_checkpoint)):
            checkpoint = torch.load(os.path.join(
                self.cp_dir, curr_checkpoint), map_location=self.device)
            cleaned_checkpoint = copy.deepcopy(checkpoint)
            if checkpoint['epoch'] == 'end' or self.start_from_next_resolution:
                # purge fade in layers from checkpoint dict so they are not loaded
                for outer_key in checkpoint.keys():
                    if type(checkpoint[outer_key]) not in [collections.OrderedDict, dict]:
                        continue
                    for key in checkpoint[outer_key].keys():
                        if "rgb" in key:
                            cleaned_checkpoint[outer_key].pop(key)
                        if "pixel" in key:
                            cleaned_checkpoint[outer_key].pop(key)
                    cleaned_checkpoint['epoch'] = 0

            self.generator.load_state_dict(
                cleaned_checkpoint['generator_state_dict'], strict=False)
            # [print(f'{k}: {v.shape}') for k, v in cleaned_checkpoint['coder_state_dict'].items()]
            self.coder.load_state_dict(
                cleaned_checkpoint['coder_state_dict'], strict=False)
            self.criticHead.load_state_dict(
                cleaned_checkpoint['critic_head_state_dict'], strict=False)
            self.coderHead.load_state_dict(
                cleaned_checkpoint['coder_head_state_dict'], strict=False)
            if load_optimizer_state:
                self.g_optimizer.load_state_dict(
                    cleaned_checkpoint['g_optimizer_state_dict'])
                self.d_optimizer.load_state_dict(
                    cleaned_checkpoint['d_optimizer_state_dict'])
                self.q_optimizer.load_state_dict(
                    cleaned_checkpoint['q_optimizer_state_dict'])
            return cleaned_checkpoint['epoch']
        else:
            print("checkpoint file not found. Starting training from scratch")
            return 0  # start epoch = 0

    def load_image_shape(self):
        """
        update image shape from checkpoint or use default if no checkpoint exists
        :return: return True if training is done
        """
        curr_checkpoint = self.get_latest_checkpoint()

        if os.path.exists(os.path.join(self.cp_dir, curr_checkpoint)):
            checkpoint = torch.load(os.path.join(
                self.cp_dir, curr_checkpoint), map_location=self.device)
            if checkpoint['epoch'] == 'end' or self.start_from_next_resolution:
                self.image_shape = (checkpoint['image_shape']
                                    [0] * 2, checkpoint['image_shape'][0] * 2)
            else:
                self.image_shape = checkpoint['image_shape']
            if self.image_shape[0] > self.max_image_shape:

                print(
                    f"Finished training max image resolution [{self.max_image_shape},{self.max_image_shape}]. Done training")
                return True
            print(f"Found checkpoint. Starting image shape={self.image_shape}")
        else:
            self.image_shape = self.start_image_shape
            print(
                f"Checkpoint file not found. Starting image shape={self.image_shape}")
            return False

    def compile(self, d_optimizer: torch.optim.Optimizer,
                g_optimizer: torch.optim.Optimizer,
                q_optimizer: torch.optim.Optimizer,
                grad_lambda: float,
                info_lambda: float):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.q_optimizer = q_optimizer
        self.grad_lambda = torch.tensor(
            grad_lambda, dtype=torch.float32, device=self.device)
        self.info_lambda = torch.tensor(
            info_lambda, dtype=torch.float32, device=self.device)

    def test_step(self, images):

        # progressive GAN alpha hyperparameter
        with torch.no_grad():
            current_epoch = torch.tensor(
                self.curr_epoch, dtype=torch.float32, device=self.device)
            alpha = torch.divide(torch.add(torch.fmod(
                current_epoch, self.epochs_per_phase), 1), self.epochs_per_phase)
            # Sample random points in the latent space
            batch_size = images.shape[0]
            size = [self.code_shape[1], self.code_shape[2]]
            channels = self.code_features + self.noise_features
            if C_AXIS == 1:
                latent_shape = [batch_size, channels, *size]
            else:
                latent_shape = [batch_size, *size, channels]
            random_latent_vectors = torch.randn(
                size=latent_shape, device=self.device)

            # Decode them to fake images
            generated_images = self.generator([random_latent_vectors, alpha])

            # gradient penalty
            eps = torch.rand([batch_size, 1, 1, 1],
                             dtype=torch.float32, device=self.device)
            interp_images = torch.mul(eps, images) + \
                torch.mul((1 - eps), generated_images)

        interp_images.requires_grad = True

        for param in chain(self.coder.parameters(), self.criticHead.parameters()):
            param.requires_grad = False
        conv_out_interp = self.coder([interp_images, alpha])
        interp_criticism = self.criticHead(conv_out_interp).sum()
        critic_x_grad = torch.autograd.grad(
            interp_criticism, interp_images, create_graph=True)
        critic_x_grad = torch.reshape(critic_x_grad[0], [batch_size, -1])
        penalty_loss = torch.mean(torch.square(
            torch.add(torch.norm(critic_x_grad, dim=-1, keepdim=True), -1)))

        for param in chain(self.coder.parameters(), self.criticHead.parameters()):
            param.requires_grad = True

        with torch.no_grad():
            # Combine them with real images
            combined_images = torch.cat([generated_images, images], dim=0)

            # Assemble labels discriminating real from fake images
            labels = torch.cat([self.fake_label * torch.ones((batch_size, 1), dtype=torch.float32, device=self.device),
                                self.real_label * torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)],
                               dim=0)

            conv_out = self.coder([combined_images, alpha])
            criticism = self.criticHead(conv_out)
            wgan_loss = torch.mean(labels * criticism)

            # Sample random points in the latent space
            random_latent_vectors = torch.randn(
                size=latent_shape, device=self.device)
            random_code = random_latent_vectors[:, :, :, :self.code_features]
            fake_images = self.generator([random_latent_vectors, alpha])
            conv = self.coder([fake_images, alpha])
            code_prediction = self.coderHead(conv)
            info_loss = torch.mean(torch.square(code_prediction - random_code))

            real_criticism_mean = torch.mean(criticism[batch_size:])
            fake_criticism_mean = torch.mean(criticism[:batch_size])
            real_criticism_std = torch.std(criticism[batch_size:])
            fake_criticism_std = torch.std(criticism[:batch_size])
            sigmoid = torch.sigmoid(criticism)
            # if torch.greater(self.real_label, self.fake_label).item():
            #     predictions = (sigmoid < 0.5).float() # sigmoid will be 0 for real 100%, so pred will be 1 here
            # else:
            #     predictions = (sigmoid < 0.5).float() # sigmoid will be 1 for real 100%, so pred will be 0 here
            predictions = (sigmoid < 0.5).float()
            critic_accuracy = (predictions == (labels / 2 + 0.5)
                               ).float().sum() / (2 * batch_size)

            losses = {"critic_loss": -wgan_loss, "info_loss": info_loss,
                      "gradient_penalty_loss": penalty_loss}
            metrics = {"critic_accuracy": critic_accuracy, "real_criticism_mean": real_criticism_mean,
                       "fake_criticism_mean": fake_criticism_mean, "real_criticism_std": real_criticism_std,
                       "fake_criticism_std": fake_criticism_std}
        return losses, metrics

    def get_clipped_grad(self, params: torch.Tensor, l=0.5, epsilon=0.001):
        if params.grad is None:
            return None
        with torch.no_grad():
            l = torch.tensor(l, dtype=torch.float32, device=self.device)
            dims = list(range(1, len(params.shape)))
            w_norm = torch.norm(params, dim=dims, keepdim=True, p=2)
            epsilon = torch.tensor(epsilon, dtype=torch.float32, device=self.device) * \
                torch.ones_like(w_norm, device=self.device)
            w_norm = torch.max(w_norm, epsilon)
            g_norm = torch.norm(params.grad, dim=dims, keepdim=True, p=2)
            cond = torch.greater(torch.divide(g_norm, w_norm), l)
            return torch.where(cond, l * torch.divide(w_norm, g_norm) * params.grad, params.grad)

    def centralized_grad(self, params: torch.Tensor):
        if params.grad is None:
            return None
        with torch.no_grad():
            dims = list(range(1, len(params.shape)))
            centers = torch.mean(params.grad, dim=dims, keepdim=True)
            return params.grad - centers

    def train_step(self, images):
        curr_epoch = torch.tensor(
            self.curr_epoch, dtype=torch.int32, device=self.device)
        # progressive GAN alpha hyper parameter
        alpha = torch.divide(torch.add(torch.fmod(
            curr_epoch, self.epochs_per_phase), 1), self.epochs_per_phase)
        # Sample random points in the latent space
        batch_size = images.shape[0]
        size = [self.code_shape[1], self.code_shape[2]]
        channels = self.code_features + self.noise_features
        if C_AXIS == 1:
            latent_shape = [batch_size, channels, *size]
        else:
            latent_shape = [batch_size, *size, channels]
        random_latent_vectors = torch.randn(
            size=latent_shape, device=self.device)

        # Decode them to fake images
        with torch.no_grad():
            generated_images = self.generator([random_latent_vectors, alpha])

            # Combine them with real images
            combined_images = torch.cat([generated_images, images], dim=0)

        # Assemble labels discriminating real from fake images
        labels = torch.cat([self.fake_label * torch.ones([batch_size, 1], device=self.device),
                            self.real_label * torch.ones([batch_size, 1], device=self.device)], dim=0)

        # Train the discriminator to optimality
        wgan_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        penalty_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        for step in range(5):
            # generate random "intermediate" images interpolating the generated and real images for gradient penalty
            eps = torch.rand(size=[batch_size, 1, 1, 1], device=self.device)
            interp_images = torch.mul(eps, images) + \
                torch.mul((1 - eps), generated_images)
            interp_images.requires_grad = True

            conv_out = self.coder([combined_images, alpha])
            criticism = self.criticHead(conv_out)
            wgan_loss = torch.mean(torch.mul(labels, criticism))

            conv_out_interp = self.coder([interp_images, alpha])
            interp_criticism = self.criticHead(conv_out_interp).sum()
            critic_x_grad = torch.autograd.grad(
                interp_criticism, interp_images, create_graph=True)
            critic_x_grad = torch.reshape(critic_x_grad[0], [batch_size, -1])
            penalty_loss = torch.mean(torch.square(
                torch.add(torch.norm(critic_x_grad, dim=-1, keepdim=True), -1)))

            d_loss = self.grad_lambda * penalty_loss + wgan_loss
            d_loss.backward()
            if self.gradient_centralization:
                for param in chain(self.coder.parameters(), self.criticHead.parameters()):
                    param.grad = self.centralized_grad(param)
            if self.adaptive_gradient_clipping:
                for param in chain(self.coder.parameters(), self.criticHead.parameters()):
                    param.grad = self.get_clipped_grad(param)
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()

            interp_images.grad.zero_()
            critic_x_grad.zero_()
        with torch.no_grad():
            real_criticism_mean = torch.mean(criticism[batch_size:])
            fake_criticism_mean = torch.mean(criticism[:batch_size])
            real_criticism_std = torch.std(criticism[batch_size:])
            fake_criticism_std = torch.std(criticism[:batch_size])
            sigmoid = torch.sigmoid(criticism)
            # if torch.greater(self.real_label, self.fake_label).item():
            #     predictions = (sigmoid < 0.5).float() # sigmoid will be 0 for real 100%, so pred will be 1 here
            # else:
            #     predictions = (sigmoid < 0.5).float() # sigmoid will be 1 for real 100%, so pred will be 0 here
            predictions = (sigmoid < 0.5).float()
            critic_accuracy = (predictions == (labels / 2 + 0.5)
                               ).float().sum() / (2 * batch_size)

        # Sample random points in the latent space
        random_latent_vectors = torch.randn(
            size=latent_shape, device=self.device)
        random_code = random_latent_vectors[:, :, :, :self.code_features]

        # Assemble labels that say "all real images"
        # This makes the generator want to create real images (match the label) since
        # we do not include an additional minus in the loss
        misleading_labels = self.real_label * \
            torch.ones([batch_size, 1], device=self.device)

        # Train the generator and encoder(note that we should *not* update the weights
        # of the critic or encoder)!
        fake_images = self.generator([random_latent_vectors, alpha])
        for param in chain(self.coder.parameters(), self.criticHead.parameters()):
            param.requires_grad = False
        conv_out_fake = self.coder([fake_images, alpha])
        fake_criticism = self.criticHead(conv_out_fake)
        for param in chain(self.coder.parameters(), self.criticHead.parameters()):
            param.requires_grad = True
        code_prediction = self.coderHead(conv_out_fake)
        g_loss = torch.mean(misleading_labels * fake_criticism)
        info_loss = torch.mean(torch.square(code_prediction - random_code))
        total_g_loss = g_loss + self.info_lambda * info_loss
        total_g_loss.backward()

        if self.adaptive_gradient_clipping:
            for param in chain(self.generator.parameters(), self.coderHead.parameters()):
                param.grad = self.get_clipped_grad(param)

        self.g_optimizer.step()
        self.q_optimizer.step()
        self.g_optimizer.zero_grad()
        self.q_optimizer.zero_grad()

        losses = {"critic_loss": -wgan_loss,
                  "info_loss": info_loss,
                  "gradient_penalty_loss": penalty_loss}
        metrics = {"critic_accuracy": critic_accuracy, "real_criticism_mean": real_criticism_mean,
                   "fake_criticism_mean": fake_criticism_mean, "real_criticism_std": real_criticism_std,
                   "fake_criticism_std": fake_criticism_std}
        return losses, metrics

    def write_tensorboard_summaries(self, real_images, global_step):
        with torch.no_grad():
            current_epoch = torch.tensor(
                self.curr_epoch, dtype=torch.float32, device=self.device)
            alpha = torch.divide(torch.add(torch.fmod(
                current_epoch, self.epochs_per_phase), 1), self.epochs_per_phase)
            # Sample random points in the latent space
            batch_size = 9
            size = [self.code_shape[1], self.code_shape[2]]
            channels = self.code_features + self.noise_features
            if C_AXIS == 1:
                latent_shape = [batch_size, channels, *size]
            else:
                latent_shape = [batch_size, *size, channels]
            random_latent_vectors = torch.randn(
                size=latent_shape, device=self.device)

            # Decode them to fake images
            generated_images = self.generator([random_latent_vectors, alpha])
            # make and save figure of images
            generated_images = torchvision.transforms.ConvertImageDtype(
                torch.uint8)(generated_images)
            img_grid = torchvision.utils.make_grid(generated_images)
            self.tensorboard_writer.add_image(
                "generated images", img_grid, global_step=global_step)

            real_images = torchvision.transforms.ConvertImageDtype(
                torch.uint8)(real_images)
            img_grid = torchvision.utils.make_grid(real_images)
            self.tensorboard_writer.add_image(
                "real images", img_grid, global_step=global_step)

            critic_params = torch.tensor(
                [], dtype=torch.float32, device=self.device)
            for param in chain(self.coder.parameters(), self.criticHead.parameters()):
                with torch.no_grad():
                    critic_params = torch.cat(
                        [critic_params, torch.flatten(param.detach())])
            critic_params = critic_params.to(device=torch.device('cpu'))
            self.tensorboard_writer.add_histogram(
                'critic params', critic_params, global_step=global_step)
            generator_params = torch.tensor(
                [], dtype=torch.float32, device=self.device)
            for param in self.generator.parameters():
                with torch.no_grad():
                    generator_params = torch.cat(
                        [generator_params, torch.flatten(param.detach())])
            generator_params = generator_params.to(device=torch.device('cpu'))
            self.tensorboard_writer.add_histogram(
                'generator params', generator_params, global_step=global_step)
