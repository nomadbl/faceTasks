import itertools
import os
import copy
import glob
import numpy.random as random_np
from sys import intern
from typing import OrderedDict
from warnings import filters
from torch._C import wait
from torch.autograd import grad
from torch.cuda import random
from torch.nn.modules.container import ModuleList
from tqdm import tqdm
from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import functools
import time
import sys
import torch

C_AXIS = 1


class Architect:
    """finite state machine to handle logic of architecture optimization"""

    def __init__(self, internal_state):
        self.handlers = {}
        self.currentState = None
        self.haltStates = []
        self.internal_state = internal_state

    def add_state(self, name, handler, haltState=False):
        name = name.upper()
        self.handlers[name] = handler
        if haltState:
            self.haltStates.append(name)

    def set_start(self, name):
        self.currentState = name.upper()

    def step(self, input):

        while True:
            try:
                handler = self.handlers[self.currentState]
            except:
                raise ValueError("must call .set_start() before .step()")
            # print(
            #     f"epoch {input['epoch']} {input['part']} entering {self.currentState}")
            # print(f'handler: {handler}')
            currentState, internal_state = handler(
                input, self.internal_state)
            # if input['part'] == 'gan':
            #     print(f"{self.currentState}->{currentState.upper()}")
            self.currentState = currentState.upper()
            self.internal_state = internal_state
            if self.currentState not in self.haltStates:
                break


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


class ResnextBranch(torch.nn.Module):
    def __init__(self, in_shape, filters, activation='relu', c_axis=C_AXIS, device=None):
        super(ResnextBranch, self).__init__()
        self.filters = filters
        self.in_ch = in_shape[c_axis]
        self.c_axis = c_axis
        if activation not in ['relu', 'leaky_relu']:
            raise ValueError(
                "Resnext block only supports relu and leaky_relu activations")
        self.activation = activation
        self.in_conv1d = torch.nn.Conv2d(
            self.in_ch, filters, kernel_size=(1, 1), device=device)
        self.conv2d = torch.nn.Conv2d(filters, filters,
                                      kernel_size=(3, 3), padding="same", device=device)
        self.out_conv1d = torch.nn.Conv2d(
            filters, self.in_ch, kernel_size=(1, 1), device=device)
        self.layer_norm = torch.nn.LayerNorm(in_shape[1:], device=device)

    def forward(self, input):
        x = input[0]
        branch_weight = input[1]
        x = self.in_conv1d(x)
        x = self.conv2d(x)
        x = self.out_conv1d(x)
        x = self.layer_norm(x)
        if self.activation == 'relu':
            x = torch.nn.ReLU()(x)
        elif self.activation == 'leaky_relu':
            x = torch.nn.LeakyReLU(negative_slope=0.1)(x)
        x = torch.mul(x, branch_weight)
        return x


class ResnextBlock(torch.nn.Module):
    def __init__(self, in_shape, filters, cardinality=1, activation='relu', c_axis=C_AXIS, device=None):
        super().__init__()
        self.cardinality = cardinality
        self.filters = filters
        self.activation = activation
        self.in_shape = in_shape
        self.c_axis = c_axis

        self.branches = torch.nn.ModuleList()
        self.branch_weights = torch.nn.ParameterList()
        for c in range(cardinality):
            self.branches.append(ResnextBranch(
                in_shape, filters, activation=activation, c_axis=c_axis, device=device))
            self.branch_weights.append(torch.nn.parameter.Parameter(
                torch.tensor(1, dtype=torch.float32, device=device), requires_grad=True))

    def forward(self, input):
        x = input[0]
        out_factor = input[1]
        for branch, branch_weight in zip(self.branches, self.branch_weights):
            x = x + out_factor * branch([input[0], branch_weight])
        return x

    def increase_cardinality(self, device=None):
        """
        increase the cardinality of this resnet block
        This is implemented for adaptive training where we increase the cardinality gradually
        """
        self.branches.append(ResnextBranch(
            self.in_shape, self.filters, activation=self.activation, c_axis=self.c_axis, device=device))
        self.branch_weights.append(torch.nn.parameter.Parameter(
            torch.tensor(0, dtype=torch.float32, device=device), requires_grad=True))

    def decrease_cardinality(self):
        """
        Reduce the cardinality of this resnet block
        This is implemented for adaptive training where we increase the cardinality gradually
        """
        def truncate(it, n):
            cache = [next(it) for i in range(n)]
            index = 0
            for val in it:
                yield cache[index]
                cache[index] = val
                index = (index + 1) % n
        modules_gen = truncate(iter(self.branches), 1)
        self.branches = torch.nn.ModuleList(iter(modules_gen))
        weights_gen = truncate(iter(self.branch_weights), 1)
        self.branch_weights = torch.nn.ParameterList(iter(weights_gen))


class AdaptiveResnext(torch.nn.Module):
    def __init__(self, in_shape, filters, activation='relu', c_axis=C_AXIS, spec=None, device=None):
        super().__init__()
        self.filters = filters
        self.activation = activation
        self.in_shape = in_shape
        self.c_axis = c_axis
        self.spec = [1] if spec is None else spec
        # spec is a list of cardianlities. one item per each block to add
        self.out_factors = torch.nn.ParameterList()
        self.blocks = torch.nn.ModuleList()
        for cardinality in self.spec:
            self.out_factors.append(torch.nn.parameter.Parameter(
                torch.tensor(1, dtype=torch.float32, device=device), requires_grad=True))
            self.blocks.append(ResnextBlock(self.in_shape, self.filters, cardinality,
                                            activation=activation, c_axis=self.c_axis, device=device))

    def forward(self, input):
        x = input
        for block, h in zip(self.blocks, self.out_factors):
            x = block([x, h])
        return x

    def increase_depth(self, device=None):
        """
        increase the depth of this resnet by 1.
        This is implemented for adaptive training where we increase the depth gradually
        """
        self.out_factors.append(torch.nn.parameter.Parameter(
            torch.tensor(0, dtype=torch.float32, device=device), requires_grad=True))
        self.blocks.append(ResnextBlock(
            self.in_shape, self.filters, activation=self.activation, c_axis=self.c_axis, device=device))
        self.spec.append(1)

    def increase_cardinality(self, block_index=-1, device=None):
        """
        return a copy of this resnet with cardinality increased by 1 on the last block.
        This is implemented for adaptive training where we increase the depth gradually
        """
        self.blocks[block_index].increase_cardinality(device=device)
        self.spec[block_index] += 1

    def decrease_cardinality(self, block_index=-1):
        """
        return a copy of this resnet with cardinality increased by 1 on the last block.
        This is implemented for adaptive training where we increase the depth gradually
        """
        self.blocks[block_index].decrease_cardinality()
        self.spec[block_index] -= 1


def new_resnext(curr_dims: DimsTracker, filters, activation, spec=None):
    resnext = AdaptiveResnext(in_shape=curr_dims.curr_dim,
                              filters=filters, activation=activation, c_axis=C_AXIS, spec=spec)
    return resnext, curr_dims


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


class Critic(torch.nn.Module):
    """
    inputs: X_i, X_(i-1)
    Compute the conditional probability P(real|X_i,X_(i-1)) and a feature code
    This implementation specializes in X being an image with varying resolution and 3 channels

    The network is a ResNext trunk operating in an embedding, whose output is used by a critics and a decoder.
    Each of them is another ResNext followed by a maxpool and conv2d respectively.
    """

    def __init__(self, input_dim: list, filters, embedding_dim, c_axis=C_AXIS, spec=None):
        super(Critic, self).__init__()
        dims = DimsTracker(input_dim)
        conditional_dim = copy.copy(dims)
        conditional_dim.update_dims(DimsTransformer(
            size_transform=lambda x: [x[0] // 2, x[1] // 2]))
        self.subsample_conv, dims = new_conv2d(dims, filters=dims.curr_channels(), stride=(
            2, 2), padding=(1, 1))  # output image size halved
        # cat here
        dims.update_dims(transform=CatTransformer(
            [dims, conditional_dim], cat_dim=c_axis))
        self.embedding, dims = new_conv2d(
            dims, filters=embedding_dim, kernel_size=(1, 1))
        self.trunk, dims = new_resnext(dims, filters, 'leaky_relu', spec=spec)
        self.toScores, dims = new_conv2d(dims, 1, kernel_size=(1, 1))
        self.mp = torch.nn.MaxPool2d(kernel_size=tuple(dims.curr_size()))

    def get_spec(self):
        return self.trunk.spec

    def increase_depth(self, device=None):
        self.trunk.increase_depth(device=device)

    def increase_cardinality(self, index, device=None):
        self.trunk.increase_cardinality(index, device=device)

    def decrease_cardinality(self, index):
        self.trunk.decrease_cardinality(index)

    def forward(self, inputs):
        x, x_prev = inputs
        x = self.subsample_conv(x)
        # skip connection
        x = torch.cat([x, x_prev], dim=C_AXIS)
        x = self.embedding(x)
        x = self.trunk(x)
        x = self.toScores(x)
        criticism = self.mp(x)
        return criticism


class Coder(torch.nn.Module):
    """
    inputs: X_i, X_(i-1)
    Compute the conditional probability P(real|X_i,X_(i-1)) and a feature code
    This implementation specializes in X being an image with varying resolution and 3 channels

    The network is a ResNext trunk operating in an embedding, whose output is used by a critic and a decoder.
    Each of them is another ResNext followed by a maxpool and conv2d respectively.
    """

    def __init__(self, input_dim: list, filters, embedding_dim, code_features, c_axis=C_AXIS, spec=None):
        super(Coder, self).__init__()
        dims = DimsTracker(input_dim)
        conditional_dim = copy.copy(dims)
        conditional_dim.update_dims(DimsTransformer(
            size_transform=lambda x: [x[0] // 2, x[1] // 2]))
        self.subsample_conv, dims = new_conv2d(dims, filters=embedding_dim, stride=(
            2, 2), padding=(1, 1))  # output image size halved
        # cat here
        dims.update_dims(transform=CatTransformer(
            [dims, conditional_dim], cat_dim=c_axis))
        self.embedding, dims = new_conv2d(
            dims, filters=embedding_dim, kernel_size=(1, 1))
        self.trunk, dims = new_resnext(dims, filters, 'leaky_relu', spec=spec)
        self.coder_layer, dims = new_conv2d(dims, code_features)
        self.code_shape = copy.copy(dims)

    def get_spec(self):
        return self.trunk.spec

    def increase_depth(self, device=None):
        self.trunk.increase_depth(device=device)

    def increase_cardinality(self, index, device=None):
        self.trunk.increase_cardinality(index, device=device)

    def decrease_cardinality(self, index):
        self.trunk.decrease_cardinality(index)

    def forward(self, inputs):
        x, x_prev = inputs
        x = self.subsample_conv(x)
        # skip connection
        x = torch.cat([x, x_prev], dim=C_AXIS)
        x = self.embedding(x)
        x = self.trunk(x)
        code = self.coder_layer(x)
        return code


class Generator(torch.nn.Module):
    def __init__(self, input_dim: list, filters, embedding_dim, c_axis=C_AXIS, spec=None):
        super(Generator, self).__init__()
        self.c_axis = c_axis
        dims = DimsTracker(input_dim)

        # doubles input size
        self.upsample_conv, dims = new_conv_trans2d(
            dims, filters=embedding_dim)
        # mix code and image with 1x1 conv
        self.embedding, dims = new_conv2d(
            dims, filters=embedding_dim, kernel_size=(1, 1))
        self.trunk, dims = new_resnext(dims, filters, 'relu', spec=spec)
        # produce image
        self.to_rgb, dims = new_conv2d(
            curr_dims=dims, filters=3, kernel_size=(1, 1))

        self.out_dims = dims

    def get_spec(self):
        return self.trunk.spec

    def increase_depth(self, device=None):
        self.trunk.increase_depth(device=device)

    def increase_cardinality(self, index, device=None):
        self.trunk.increase_cardinality(index, device=device)

    def decrease_cardinality(self, index):
        self.trunk.decrease_cardinality(index)

    def forward(self, inputs):
        if self.c_axis == 1:
            prev_image = inputs[:, :, :, :]
        else:
            prev_image = inputs[:, :, :, :]
        x = self.upsample_conv(prev_image)
        x = self.embedding(x)
        x = self.trunk(x)
        x = self.to_rgb(x)
        x = torch.tanh(x)  # range [-1, 1]
        x = torch.mul(x, 1/2)  # range [-1/2, 1/2]
        x = torch.add(x, 1/2)  # range [0, 1]
        return x


class GeneratorCollection(torch.nn.Module):
    def __init__(self, batch_size, code_features, noise_features, final_image_shape, embedding_dim,
                 eval_model, specs=None):
        super(GeneratorCollection, self).__init__()
        self.final_image_shape = final_image_shape
        self.eval_model = eval_model
        self.code_features = code_features
        self.noise_features = noise_features
        self.channels = 3 + code_features + noise_features
        size = [2, 2]
        if C_AXIS == 1:
            inp_dims = DimsTracker(
                input_dim=[batch_size, self.channels, *size])
        else:
            inp_dims = DimsTracker(
                input_dim=[batch_size, *size, self.channels])
        print("building generators")
        print("-------------------")
        print(f"layer       in_shape           out_shape")
        self.generators = torch.nn.ModuleList()
        dims = inp_dims
        self.index_to_size = []
        i = 0
        while size != final_image_shape:
            size = [size[0] * 2, size[1] * 2]
            self.index_to_size.append(tuple(size))
            spec = None if specs is None else specs[tuple(
                size)]
            self.generators.append(
                Generator(dims.curr_dim, filters=4, embedding_dim=embedding_dim, spec=spec))
            dims.update_dims(
                transform=lambda _: self.generators[-1].out_dims.curr_dim)
            print(
                f"block_{i+1}        {dims.prev_dim}          {dims.curr_dim}")
            if tuple(dims.curr_size()) == self.final_image_shape:
                break
            # input of all blocks is always "channels" channels
            dims.update_dims(
                transform=DimsTransformer(channels_transform=lambda x: self.channels))
            i += 1

        print("\n")
        if eval_model:
            self.eval()

        self.set_training_index(-1)

    def get_specs(self):
        specs = {self.index_to_size[i]: self.generators[i].get_spec()
                 for i in range(len(self.generators))}
        return specs

    def set_training_index(self, index: -1):
        self.current_training_index = 0 if index < 0 else index
        self.out_dims = self.generators[index].out_dims

    def increase_depth(self, device=None):
        self.generators[self.current_training_index].increase_depth(
            device=device)

    def increase_cardinality(self, index, device=None):
        self.generators[self.current_training_index].increase_cardinality(
            index, device=device)

    def decrease_cardinality(self, index):
        self.generators[self.current_training_index].decrease_cardinality(
            index)

    def forward(self, inputs):
        if self.eval_model:
            # [code0, code1...] = inputs.
            if C_AXIS == 1:
                x = torch.zeros(inputs[0].shape[0], 3, *inputs[0].shape[2:])
            else:
                x = torch.zeros(inputs[0].shape[0], *inputs[0].shape[1:3], 3)
            output = [x]  # include initial code and zeros
            for generator, code in zip(self.generators, inputs):
                x = torch.cat([x, code], dim=C_AXIS)
                x = generator(x)
                output.append(x)
            return output
        else:
            # code = inputs. x already appended to code
            x = self.generators[self.current_training_index](inputs)
            return x


class CriticCollection(torch.nn.Module):
    """
    Conditional discriminator and decoder
    criticism, codes = critic(x)
    where the conditional criticism is for the specified image shape and the
    codes are the concatenation of all the codes.
    """

    def __init__(self, batch_size, embedding_dim, final_image_shape,
                 eval_model, specs=None):
        super(CriticCollection, self).__init__()
        print("building critics")
        print("-------------------")
        print(f"critic           in_shape")

        self.critics = torch.nn.ModuleList()
        size = [2, 2]
        self.index_to_size = []
        i = 0
        while size != final_image_shape:
            size = [size[0] * 2, size[1] * 2]
            spec = None if specs is None else specs[tuple(
                size)]
            self.index_to_size.append(tuple(size))
            input_dim = [batch_size, 3, *
                         size] if C_AXIS == 1 else [batch_size, *size, 3]
            self.critics.append(
                Critic(input_dim, filters=4, embedding_dim=embedding_dim, spec=spec))
            print(f"block_{i+1}     {input_dim}")
            if tuple(size) == final_image_shape:
                break
            i += 1

        print("\n")
        if eval_model:
            self.eval()
        self.eval_model = eval_model
        self.set_training_index(0)

    def set_training_index(self, index):
        self.current_training_index = index

    def increase_depth(self, device=None):
        self.critics[self.current_training_index].increase_depth(device=device)

    def increase_cardinality(self, index, device=None):
        self.critics[self.current_training_index].increase_cardinality(
            index, device=device)

    def decrease_cardinality(self, index):
        self.critics[self.current_training_index].decrease_cardinality(index)

    def get_specs(self):
        specs = {self.index_to_size[i]: self.critics[i].get_spec()
                 for i in range(len(self.critics))}
        return specs

    def forward(self, inputs):
        if self.eval_model:
            # at evaluation, get all codes and throw out the criticism signal
            # [zeros, x_0, x_1, x_2...] = inputs
            criticisms = []
            for i in range(len(self.critics)):
                x_prev = inputs[i]
                x = inputs[i+1]
                criticisms.append(self.critics[i]([x, x_prev]))
            return criticisms
        else:
            # x, xprev = inputs
            # at training only use the relavant block
            criticism = self.critics[self.current_training_index](
                inputs)
            return criticism


class CoderCollection(torch.nn.Module):
    """
    Conditional coders
    """

    def __init__(self, batch_size, embedding_dim, code_features, final_image_shape,
                 eval_model, specs=None):
        super(CoderCollection, self).__init__()
        print("building coders")
        print("-------------------")
        print(f"coder           in_shape")

        self.code_features = code_features
        self.coders = torch.nn.ModuleList()
        size = [2, 2]
        self.index_to_size = []
        i = 0
        while size != final_image_shape:
            size = [size[0] * 2, size[1] * 2]
            self.index_to_size.append(tuple(size))
            spec = None if specs is None else specs[tuple(size)]
            input_dim = [batch_size, 3, *
                         size] if C_AXIS == 1 else [batch_size, *size, 3]
            self.coders.append(
                Coder(input_dim, filters=4, embedding_dim=embedding_dim, code_features=code_features, spec=spec))
            print(f"block_{i+1}     {input_dim}")
            if tuple(size) == final_image_shape:
                break
            i += 1

        print("\n")
        if eval_model:
            self.eval()
        self.eval_model = eval_model
        self.set_training_index(0)

    def set_training_index(self, index):
        self.current_training_index = index

    def get_specs(self):
        specs = {self.index_to_size[i]: self.coders[i].get_spec()
                 for i in range(len(self.coders))}
        return specs

    def increase_depth(self, device=None):
        self.coders[self.current_training_index].increase_depth(device=device)

    def increase_cardinality(self, index, device=None):
        self.coders[self.current_training_index].increase_cardinality(
            index, device=device)

    def decrease_cardinality(self, index):
        self.coders[self.current_training_index].decrease_cardinality(index)

    def forward(self, inputs):
        if self.eval_model:
            # [zeros, x_0, x_1, x_2...] = inputs
            codes = []
            for i in range(len(self.coders)):
                x_prev = inputs[i]
                x = inputs[i+1]
                codes.append(self.coders[i]([x, x_prev]))
            return codes
        else:
            # x, xprev = inputs
            # at training only use the relavant block
            code = self.coders[self.current_training_index](
                inputs)
            return code


class FacesDataset(Dataset):
    def __init__(self, files, start_size=(4, 4), size=(128, 128)):
        self.files = files
        self.ln = len(files)
        self.size = size
        self.prev_size = (size[0] // 2, size[1] // 2)
        self.start_size = start_size

    def update_size(self, size):
        self.prev_size = (size[0] // 2, size[1] // 2)
        self.size = size

    def __getitem__(self, item):
        file = self.files[item]
        file = torchvision.io.read_file(file)
        image = torchvision.io.decode_jpeg(file)
        image = torchvision.transforms.Resize(size=self.size)(image)
        if self.prev_size is not self.start_size:
            prev_image = torchvision.transforms.Resize(
                size=self.prev_size)(image)
        else:
            prev_image = torch.zeros_like(image)
            prev_image = torchvision.transforms.Resize(
                size=self.prev_size)(prev_image)
        return torchvision.transforms.ConvertImageDtype(torch.float32)(image), torchvision.transforms.ConvertImageDtype(torch.float32)(prev_image)

    def __len__(self):
        return self.ln


class AdaptiveStageGAN:
    def __init__(self, files_dir, code_features, noise_features,
                 embedding_dim, cp_dir, epochs_per_phase=5, batch_sizes=None,
                 info_lambda=100,
                 grad_lambda=10,
                 lr=0.001,
                 adaptive_gradient_clipping=False,
                 gradient_centralization=False,
                 start_image_shape=(4, 4),
                 final_image_shape=(128, 128),
                 start_from_next_resolution=False,
                 specs=None,
                 debug_architect=False):
        super(AdaptiveStageGAN, self).__init__()

        self.final_image_shape = final_image_shape
        self.start_image_shape = start_image_shape
        self.start_from_next_resolution = start_from_next_resolution
        self.image_shape = start_image_shape
        self.noise_features = noise_features
        self.code_features = code_features
        self.embedding_dim = embedding_dim

        self.batch_sizes = {(4, 4): 256,
                            (8, 8): 256,
                            (16, 16): 128,
                            (32, 32): 128,
                            (64, 64): 128,
                            (128, 128): 64}
        if type(batch_sizes) is int:
            for key in self.batch_sizes.keys():
                self.batch_sizes[key] = batch_sizes

        if specs is None:
            init_specs = {(4, 4): None,
                          (8, 8): None,
                          (16, 16): None,
                          (32, 32): None,
                          (64, 64): None,
                          (128, 128): None}
            self.specs = {"generator_specs": init_specs,
                          "critic_specs": init_specs,
                          "coder_specs": init_specs}
        else:
            self.specs = specs

        tmp_image_shape = start_image_shape
        self.training_indices = {}
        ind = 0
        while tmp_image_shape[0] <= final_image_shape[0]:
            self.training_indices[tmp_image_shape] = ind
            tmp_image_shape = (
                2 * tmp_image_shape[0], 2 * tmp_image_shape[1])
            ind += 1

        self.cp_dir = cp_dir
        if not os.path.exists(cp_dir):
            os.mkdir(cp_dir)

        self.tensorboard_writer = None

        self.epochs_per_phase = epochs_per_phase
        self.curr_epoch = 0

        self.real_label = torch.tensor(-1, dtype=torch.float32)
        self.fake_label = torch.tensor(1, dtype=torch.float32)

        self.d_optimizer = None
        self.g_optimizer = None
        self.q_optimizer = None
        self.grad_lambda = torch.tensor(info_lambda)
        self.info_lambda = torch.tensor(grad_lambda)
        self.generators = torch.nn.Module()
        self.critics = torch.nn.Module()
        self.coders = torch.nn.Module()

        self.lr = lr
        self.adaptive_gradient_clipping = adaptive_gradient_clipping
        self.gradient_centralization = gradient_centralization

        self.updated_gan_cardinality_flag = False
        self.updated_coder_cardinality_flag = False

        self.files_dir = files_dir

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.debug_architect = debug_architect

    def build_models(self, batchl_size, eval_model=False):
        """
        initialize models and send to device
        :param image_shape: current image shape in pGAN training
        :param eval_model: return evaluation models if True. defaults to False
        :return: None
        """
        specs = self.specs["generator_specs"]
        self.generators = GeneratorCollection(
            batchl_size, self.code_features, self.noise_features, self.final_image_shape, self.embedding_dim, eval_model, specs)
        specs = self.specs["critic_specs"]
        self.critics = CriticCollection(
            batchl_size, self.embedding_dim, self.final_image_shape, eval_model, specs)
        specs = self.specs["coder_specs"]
        self.coders = CoderCollection(
            batchl_size, self.embedding_dim, self.code_features, self.final_image_shape, eval_model, specs)
        if not eval_model:
            self.generators.set_training_index(
                self.training_indices[self.image_shape])
            self.critics.set_training_index(
                self.training_indices[self.image_shape])
            self.coders.set_training_index(
                self.training_indices[self.image_shape])
        self.updated_gan_cardinality_flag = False
        self.updated_coder_cardinality_flag = False

    def models_to_device(self):
        self.generators.to(self.device)
        self.critics.to(self.device)
        self.coders.to(self.device)

    def get_image_dataset(self, files_dir):
        files = glob.glob(files_dir)
        image_count = len(files)
        # train/test split
        train_percent = 0.8
        train_samples = int(round(image_count * train_percent))
        train_files = files[:train_samples]
        val_files = files[train_samples:]
        train_dataset = FacesDataset(
            train_files, size=self.image_shape)
        val_dataset = FacesDataset(
            val_files, size=self.image_shape)
        return train_dataset, val_dataset

    def process_epoch(self, train_ds, eval_ds, epoch):

        if self.debug_architect:
            # key = input("press q to quit, anything else to continue")
            # if key == 'q':
            #     sys.exit()
            r_loss = torch.rand(100, 8).sum(dim=1)
            r_loss = torch.tensor(epoch+1) * r_loss
            # print(f"epoch={epoch}")
            losses = {"critic_loss": r_loss[0].item(),
                      "info_loss": r_loss[1].item(),
                      "gradient_penalty_loss": r_loss[2].item()}
            metrics = {"critic_accuracy": r_loss[3].item(), "real_criticism_mean": r_loss[4].item(),
                       "fake_criticism_mean": r_loss[5].item(), "real_criticism_std": r_loss[6].item(),
                       "fake_criticism_std": r_loss[7].item()}
            last_running = losses
            last_running.update(metrics)
            last_eval_running = losses
            last_eval_running.update(metrics)
            return losses, metrics, last_running, last_eval_running

        test_steps = 5
        # get datasets of resized images
        pin_memory = torch.cuda.is_available()
        num_workers = 4 if torch.cuda.is_available() else 0
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_sizes[self.image_shape], shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        eval_loader = DataLoader(
            eval_ds, batch_size=self.batch_sizes[self.image_shape], shuffle=True)
        n_total_steps = len(train_loader)
        progress_bar = tqdm(train_loader)
        description = f"Epoch {epoch}"
        progress_bar.set_description(description)
        i = 0
        running = {}
        eval_running = {}
        for batch in progress_bar:
            i += 1
            batch[0] = batch[0].to(self.device)
            batch[1] = batch[1].to(self.device)
            losses, metrics = self.train_step(batch)
            global_step = epoch * n_total_steps + i
            for scalar, value in chain(iter(losses.items()), iter(metrics.items())):
                if scalar not in running.keys():
                    running[scalar] = 0.0
                running[scalar] += value.item()
            running_steps = n_total_steps // 4
            if (i + 1) % running_steps == 0:
                # test steps
                it = iter(eval_loader)
                for step in range(test_steps):
                    images = next(it)
                    test_images = [images[0].to(
                        self.device), images[1].to(self.device)]
                    eval_losses, eval_metrics = self.test_step(
                        test_images)
                    for scalar, value in chain(iter(eval_losses.items()), iter(eval_metrics.items())):
                        if scalar not in eval_running.keys():
                            eval_running[scalar] = 0.0
                        eval_running[scalar] += value.item()
                last_running = copy.copy(running)
                last_eval_running = copy.copy(eval_running)
                self.write_tensorboard_summaries(
                    batch, running, eval_running, running_steps, global_step)
                running = {}
                eval_running = {}
        progress_bar.close()
        return losses, metrics, last_running, last_eval_running

    def increase_coder_depth(self, input, internal_state):
        self.coders.increase_depth(device=self.device)
        internal_state["cardinality"][self.image_shape].append(1)
        internal_state["index"] = 0
        internal_state["any_change"] = True
        self.specs["coder_specs"] = self.coders.get_specs()
        new_state = "train"
        return new_state, internal_state

    def increase_gan_depth(self, input, internal_state):
        self.generators.increase_depth(device=self.device)
        self.critics.increase_depth(device=self.device)
        self.specs["generator_specs"] = self.generators.get_specs()
        self.specs["critic_specs"] = self.critics.get_specs()
        internal_state["cardinality"][self.image_shape].append(1)
        internal_state["index"] = 0
        internal_state["any_change"] = True
        new_state = "train"
        return new_state, internal_state

    def decrease_gan_cardinality(self, input, internal_state):
        index = internal_state["index"]
        internal_state["updated_cardinality"] = False
        if internal_state["cardinality"][self.image_shape] is None:
            internal_state["cardinality"][self.image_shape] = 1
        if internal_state["cardinality"][self.image_shape][index] > 1:
            self.generators.decrease_cardinality(index)
            self.critics.decrease_cardinality(index)
            self.specs["generator_specs"] = self.generators.get_specs()
            self.specs["critic_specs"] = self.critics.get_specs()
            internal_state["any_change"] = True
            self.load_cp(get_backup=True, part='gan')
            internal_state["cardinality"][self.image_shape][index] -= 1
        if internal_state["index"] == len(internal_state["cardinality"][self.image_shape]) - 1:
            new_state = "increase_depth"
        else:
            internal_state["index"] += 1
            new_state = "train"
        return new_state, internal_state

    def increase_gan_cardinality(self, input, internal_state):
        index = internal_state["index"]
        self.generators.increase_cardinality(index, device=self.device)
        self.critics.increase_cardinality(index, device=self.device)

        assert self.generators.get_specs() == self.critics.get_specs()

        self.specs["generator_specs"] = copy.deepcopy(
            self.generators.get_specs())
        self.specs["critic_specs"] = copy.deepcopy(self.critics.get_specs())

        internal_state["updated_cardinality"] = True
        internal_state["any_change"] = True
        if internal_state["cardinality"][tuple(self.image_shape)] is None:
            internal_state["cardinality"][tuple(self.image_shape)] = [1]
        internal_state["cardinality"][tuple(self.image_shape)][index] += 1
        new_state = "train"
        return new_state, internal_state

    def decrease_coder_cardinality(self, input, internal_state):
        index = internal_state["index"]
        internal_state["updated_cardinality"] = False
        if internal_state["cardinality"][self.image_shape] is None:
            internal_state["cardinality"][self.image_shape] = 1
        if internal_state["cardinality"][self.image_shape][index] > 1:
            self.coders.decrease_cardinality(index)
            internal_state["any_change"] = True
            self.specs["coder_specs"] = self.coders.get_specs()
            self.load_cp(get_backup=True, part="coder")
            internal_state["cardinality"][self.image_shape][index] -= 1
        if internal_state["index"] == len(internal_state["cardinality"][self.image_shape]) - 1:
            new_state = "increase_depth"
        else:
            internal_state["index"] += 1
            new_state = "train"
        return new_state, internal_state

    def increase_coder_cardinality(self, input, internal_state):
        index = internal_state["index"]
        self.coders.increase_cardinality(index, device=self.device)
        self.specs["coder_specs"] = self.coders.get_specs()
        internal_state["updated_cardinality"] = True
        internal_state["any_change"] = True
        if internal_state["cardinality"][tuple(self.image_shape)] is None:
            internal_state["cardinality"][tuple(self.image_shape)] = [1]
        internal_state["cardinality"][tuple(self.image_shape)][index] += 1
        new_state = "train"
        return new_state, internal_state

    def architect_prepare_train(self, input, internal_state):
        if internal_state["any_change"]:
            self.compile(d_optimizer=torch.optim.Adam(self.critics.parameters(), lr=self.lr),
                         g_optimizer=torch.optim.Adam(
                             self.generators.parameters(), lr=self.lr),
                         q_optimizer=torch.optim.Adam(
                             self.coders.parameters(), lr=self.lr),
                         grad_lambda=10, info_lambda=0.01)
            self.models_to_device()
            internal_state["any_change"] = False
        next_state = "check_loss"
        return next_state, internal_state

    def architect_check_loss(self, input, internal_state):
        epoch = input["epoch"]
        loss_name = input["loss_name"]
        part = input["part"]
        losses = input["losses"]
        running = input["running"]
        threshold = input["threshold"]

        loss = running[loss_name]
        prev_loss = internal_state["loss"]
        d_loss = abs(loss - prev_loss) / abs(loss)
        if d_loss < threshold:
            if internal_state["updated_cardinality"]:
                next_state = "decrease_cardinality"
            else:
                next_state = "increase_cardinality"
        else:
            self.save_cp(epoch=epoch, losses=losses, backup=True, part=part)
            internal_state["updated_cardinality"] = False
            next_state = "train"

        internal_state["loss"] = loss

        return next_state, internal_state

    def init_architects(self):
        gan_architect = Architect(internal_state={"any_change": False,
                                                  "updated_cardinality": False,
                                                  "cardinality": copy.deepcopy(self.generators.get_specs()),
                                                  "index": 0,
                                                  "loss": float("inf")})
        coder_architect = Architect(internal_state={"any_change": False,
                                                    "updated_cardinality": False,
                                                    "cardinality": copy.deepcopy(self.coders.get_specs()),
                                                    "index": 0,
                                                    "loss": float("inf")})

        def reset_architect_index(input, internal_state):
            internal_state["index"] = 0
            return ("check_loss", internal_state)

        gan_architect.add_state(
            name="train", handler=self.architect_prepare_train)
        gan_architect.add_state(
            name="check_loss", handler=self.architect_check_loss, haltState=True)
        gan_architect.add_state(
            name="increase_cardinality", handler=self.increase_gan_cardinality)
        gan_architect.add_state(
            name="decrease_cardinality", handler=self.decrease_gan_cardinality)
        gan_architect.add_state(
            name="increase_depth", handler=self.increase_gan_depth)
        gan_architect.add_state(
            name="reset_index", handler=reset_architect_index)

        coder_architect.add_state(
            name="train", handler=self.architect_prepare_train)
        coder_architect.add_state(
            name="check_loss", handler=self.architect_check_loss, haltState=True)
        coder_architect.add_state(
            name="increase_cardinality", handler=self.increase_coder_cardinality)
        coder_architect.add_state(
            name="decrease_cardinality", handler=self.decrease_coder_cardinality)
        coder_architect.add_state(
            name="increase_depth", handler=self.increase_coder_depth)
        coder_architect.add_state(
            name="reset_index", handler=reset_architect_index)

        gan_architect.set_start("train")
        coder_architect.set_start("train")
        return gan_architect, coder_architect

    def fit(self):
        """
        Custom training loop.
        load models as needed.
        Adjust image inputs to correct resolution
        :return:
        """
        while True:

            # build models
            finished = self.load_image_shape()
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.cp_dir, f"summaries_{self.image_shape[0]}_{self.image_shape[1]}"))
            if finished:
                break
            print(f"learning rate = {self.lr}")
            train_ds, eval_ds = self.get_image_dataset(self.files_dir)
            batchl_size = self.batch_sizes[self.image_shape]

            self.load_cp(get_specs=True)
            self.build_models(batchl_size, eval_model=False)

            # define architect logic
            gan_architect, coder_architect = self.init_architects()

            self.compile(d_optimizer=torch.optim.Adam(self.critics.parameters(), lr=self.lr),
                         g_optimizer=torch.optim.Adam(
                             self.generators.parameters(), lr=self.lr),
                         q_optimizer=torch.optim.Adam(
                             self.coders.parameters(), lr=self.lr),
                         grad_lambda=10, info_lambda=0.01)
            # load checkpoint
            start_epoch, gan_architect_internal_state = self.load_cp(
                part="gan")
            _, coder_architect_internal_state = self.load_cp(part="coder")

            if gan_architect_internal_state:
                gan_architect.internal_state = gan_architect_internal_state
            if coder_architect_internal_state:
                coder_architect.internal_state = coder_architect_internal_state

            if self.start_from_next_resolution:
                self.start_from_next_resolution = False  # only do once

            losses = None
            running = None
            for epoch in range(start_epoch, self.epochs_per_phase):
                print("current gan architecture")
                print("------------------------")
                print(
                    f"generator spec: {self.specs['generator_specs'][self.image_shape]}")
                print(
                    f"critic spec: {self.specs['critic_specs'][self.image_shape]}")
                print(
                    f"coder spec: {self.specs['coder_specs'][self.image_shape]}")
                # train epoch and get running metrics
                if epoch == 0:
                    self.models_to_device()

                losses, metrics, running, eval_running = self.process_epoch(
                    train_ds, eval_ds, epoch)

                # optimize architectures
                threshold = 0.1
                loss_name = "critic_loss"
                part = "gan"
                architect_input = {"epoch": epoch, "loss_name": loss_name,
                                   "part": part, "losses": losses, "running": running,
                                   "threshold": threshold}
                gan_architect.step(architect_input)

                assert gan_architect.internal_state["cardinality"] == self.generators.get_specs(
                ), "specs dont match"
                assert gan_architect.internal_state["cardinality"] == self.critics.get_specs(
                ), "specs dont match"

                loss_name = "info_loss"
                part = "coder"
                architect_input = {"epoch": epoch, "loss_name": loss_name,
                                   "part": part, "losses": losses, "running": running,
                                   "threshold": threshold}
                coder_architect.step(architect_input)

                if epoch % (self.epochs_per_phase // 4) == (self.epochs_per_phase // 4) - 1:
                    self.save_cp(epoch, losses, part="gan",
                                 architect=gan_architect)
                    self.save_cp(epoch, losses, part="coder",
                                 architect=coder_architect)

            # checkpoint model
            self.save_cp('end', losses, part="gan", architect=gan_architect)
            self.save_cp('end', losses, part="coder",
                         architect=coder_architect)

        self.tensorboard_writer.close()

    def save_cp(self, epoch, losses, backup=False, part='gan', architect=None):
        if part == 'gan':
            cp_dict = {
                'epoch': epoch,
                'generator_state_dict': self.generators.state_dict(),
                'critic_state_dict': self.critics.state_dict(),
                'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                'image_shape': self.image_shape,
                'generator_specs': self.specs['generator_specs'],
                'critic_specs': self.specs['critic_specs']}
        elif part == 'coder':
            cp_dict = {
                'epoch': epoch,
                'coder_state_dict': self.coders.state_dict(),
                'q_optimizer_state_dict': self.q_optimizer.state_dict(),
                'image_shape': self.image_shape,
                'coder_specs': self.specs['coder_specs']}

        cp_dict.update(
            {"architect_state": architect.internal_state if architect else None})

        cp_dict.update(losses)
        if not os.path.exists(self.cp_dir):
            os.mkdir(self.cp_dir)
        pre = f'backup_{part}_checkpoint' if backup else f'{part}_checkpoint'
        torch.save(cp_dict, os.path.join(
            self.cp_dir, f"{pre}_{self.image_shape[0]}_{self.image_shape[1]}.pth"))

    def get_latest_checkpoint(self, backup=False, part='gan'):
        # default value
        pre = f"backup_{part}_checkpoint" if backup else f"{part}_checkpoint"
        curr_checkpoint = f"{pre}_{self.start_image_shape[0]}_{self.start_image_shape[1]}.pth"
        if os.path.exists(self.cp_dir):
            # find latest checkpoint

            checkpoints = glob.glob(os.path.join(
                self.cp_dir, "{pre}_*_*.pth"))
            if len(checkpoints) > 0:
                def get_image_shape_from_cp(cp):
                    fn = cp.split(sep='/')[-1]
                    fn = fn.split(sep='.')[0]
                    i = fn.split(sep='_')[-2]
                    return int(i)
                checkpoints_by_image_shape = {
                    get_image_shape_from_cp(cp): cp for cp in checkpoints}
                max_image_shape = max(checkpoints_by_image_shape.keys())
                curr_checkpoint = checkpoints_by_image_shape[max_image_shape]

        return curr_checkpoint

    def load_cp(self, load_optimizer_state=False, get_specs=False, get_backup=False, part="gan"):
        curr_checkpoint = self.get_latest_checkpoint(get_backup, part)
        print(f"current checkpoint {curr_checkpoint}")
        if os.path.exists(os.path.join(self.cp_dir, curr_checkpoint)):
            checkpoint = torch.load(os.path.join(
                self.cp_dir, curr_checkpoint), map_location=self.device)

            if get_specs:
                if part == "gan":
                    self.specs['generator_specs'] = checkpoint['generator_specs']
                    self.specs['critic_specs'] = checkpoint['critic_specs']
                elif part == "coder":
                    self.specs["coder_specs"] = checkpoint['coder_specs']
                return

            cleaned_checkpoint = copy.deepcopy(checkpoint)
            if checkpoint['epoch'] == 'end' or self.start_from_next_resolution:
                cleaned_checkpoint['epoch'] = 0
                cleaned_checkpoint["architect_state"] = None

            if part == "gan":
                self.generators.load_state_dict(
                    cleaned_checkpoint['generator_state_dict'], strict=False)
                self.critics.load_state_dict(
                    cleaned_checkpoint['critic_state_dict'], strict=False)
                if load_optimizer_state:
                    self.g_optimizer.load_state_dict(
                        cleaned_checkpoint['g_optimizer_state_dict'])
                    self.d_optimizer.load_state_dict(
                        cleaned_checkpoint['d_optimizer_state_dict'])
            elif part == "coder":
                self.coders.load_state_dict(
                    cleaned_checkpoint['coder_state_dict'], strict=False)
                if load_optimizer_state:
                    self.q_optimizer.load_state_dict(
                        cleaned_checkpoint['q_optimizer_state_dict'])
            return cleaned_checkpoint['epoch'], cleaned_checkpoint["architect_state"]
        else:
            print(
                "checkpoint file not found. returning epoch = 0 (start training from scratch?)")
            return 0, None  # start epoch = 0

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
            if self.image_shape[0] > self.final_image_shape[0]:

                print(
                    f"Finished training max image resolution [{self.final_image_shape},{self.final_image_shape}]. Done training")
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
        prev_images = images[1]
        curr_images = images[0]
        with torch.no_grad():
            # Sample random points in the latent space
            batch_size = curr_images.shape[0]
            channels = self.code_features + self.noise_features
            if C_AXIS == 1:
                size = [prev_images.shape[2], prev_images.shape[3]]
                latent_shape = [batch_size, channels, *size]
            else:
                size = [prev_images.shape[1], prev_images.shape[2]]
                latent_shape = [batch_size, *size, channels]
            random_latent_vectors = torch.randn(
                size=latent_shape, device=self.device)
            coded_prev_image = torch.cat(
                [prev_images, random_latent_vectors], dim=C_AXIS)
            # Decode them to fake images
            generated_images = self.generators(coded_prev_image)

            # gradient penalty
            eps = torch.rand([batch_size, 1, 1, 1],
                             dtype=torch.float32, device=self.device)
            interp_images = torch.mul(eps, curr_images) + \
                torch.mul((1 - eps), generated_images)

        interp_images.requires_grad = True

        for param in self.critics.parameters():
            param.requires_grad = False
        interp_criticism = self.critics([interp_images, prev_images])
        interp_criticism = interp_criticism.sum()
        critic_x_grad = torch.autograd.grad(
            interp_criticism, interp_images, create_graph=True)
        critic_x_grad = torch.reshape(critic_x_grad[0], [batch_size, -1])
        penalty_loss = torch.mean(torch.square(
            torch.add(torch.norm(critic_x_grad, dim=-1, keepdim=True), -1)))

        for param in self.critics.parameters():
            param.requires_grad = True

        with torch.no_grad():
            # Combine them with real images
            combined_images = torch.cat([generated_images, curr_images], dim=0)
            combined_prev_images = torch.cat([prev_images, prev_images], dim=0)

            # Assemble labels discriminating real from fake images
            labels = torch.cat([self.fake_label * torch.ones((batch_size, 1), dtype=torch.float32, device=self.device),
                                self.real_label * torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)],
                               dim=0)
            labels = labels.reshape([2*batch_size, 1, 1, 1])

            criticism = self.critics(
                [combined_images, combined_prev_images])
            code_prediction = self.coders(
                [combined_images, combined_prev_images])
            # retain only code with corresponding original code
            code_prediction = code_prediction[:batch_size]
            wgan_loss = torch.mean(labels * criticism)

            if C_AXIS == 1:
                random_code = random_latent_vectors[:,
                                                    :self.code_features, :, :]
            else:
                random_code = random_latent_vectors[:,
                                                    :, :, :self.code_features]
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
        # Sample random points in the latent space
        curr_images = images[0]
        prev_images = images[1]
        batch_size = curr_images.shape[0]

        channels = self.code_features + self.noise_features
        if C_AXIS == 1:
            size = [prev_images.shape[2], prev_images.shape[3]]
            latent_shape = [batch_size, channels, *size]
        else:
            size = [prev_images.shape[1], prev_images.shape[2]]
            latent_shape = [batch_size, *size, channels]
        random_latent_vectors = torch.randn(
            size=latent_shape, device=self.device)

        if self.image_shape == (4, 4):
            generator_input = torch.cat(
                [torch.zeros_like(prev_images), random_latent_vectors], dim=C_AXIS)
        else:
            generator_input = torch.cat(
                [prev_images, random_latent_vectors], dim=C_AXIS)
        # Decode them to fake images
        with torch.no_grad():
            generated_images = self.generators(generator_input)

        # Combine them with real images
        combined_images = torch.cat([generated_images, curr_images], dim=0)
        combined_prev_images = torch.cat([prev_images, prev_images], dim=0)

        # Assemble labels discriminating real from fake images
        labels = torch.cat([self.fake_label * torch.ones([batch_size, 1], device=self.device),
                            self.real_label * torch.ones([batch_size, 1], device=self.device)], dim=0)
        labels = labels.reshape([2*batch_size, 1, 1, 1])

        # Train the discriminator to optimality
        # wgan_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        # penalty_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        for step in range(5):
            # generate random "intermediate" images interpolating the generated and real images for gradient penalty
            eps = torch.rand(size=[batch_size, 1, 1, 1], device=self.device)
            interp_images = torch.mul(eps, curr_images) + \
                torch.mul((1 - eps), generated_images)
            interp_images.requires_grad = True

            criticism = self.critics(
                [combined_images, combined_prev_images])
            wgan_loss = torch.mean(torch.mul(labels, criticism))

            interp_criticism = self.critics([interp_images, prev_images])
            interp_criticism = interp_criticism.sum()
            critic_x_grad = torch.autograd.grad(
                interp_criticism, interp_images, create_graph=True)
            critic_x_grad = torch.reshape(critic_x_grad[0], [batch_size, -1])
            penalty_loss = torch.mean(torch.square(
                torch.add(torch.norm(critic_x_grad, dim=-1, keepdim=True), -1)))

            d_loss = self.grad_lambda * penalty_loss + wgan_loss
            d_loss.backward()
            if self.gradient_centralization:
                for param in self.critics.parameters():
                    param.grad = self.centralized_grad(param)
            if self.adaptive_gradient_clipping:
                for param in self.critics.parameters():
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

        if C_AXIS == 1:
            random_code = random_latent_vectors[:, :self.code_features, :, :]
        else:
            random_code = random_latent_vectors[:, :, :, :self.code_features]

        if self.image_shape == (4, 4):
            generator_input = torch.cat(
                [torch.zeros_like(prev_images), random_latent_vectors], dim=C_AXIS)
        else:
            generator_input = torch.cat(
                [prev_images, random_latent_vectors], dim=C_AXIS)
        # Assemble labels that say "all real images"
        # This makes the generators want to create real images (match the label) since
        # we do not include an additional minus in the loss
        misleading_labels = self.real_label * \
            torch.ones([batch_size, 1, 1, 1], device=self.device)

        # Train the generators and encoder(note that we should *not* update the weights
        # of the critics)!
        fake_images = self.generators(generator_input)

        for param in self.critics.parameters():
            param.requires_grad = False
        fake_criticism = self.critics(
            [fake_images, prev_images])
        code_prediction = self.coders([fake_images, prev_images])
        for param in self.critics.parameters():
            param.requires_grad = True
        g_loss = torch.mean(misleading_labels * fake_criticism)
        info_loss = torch.mean(torch.square(code_prediction - random_code))
        total_g_loss = g_loss + self.info_lambda * info_loss
        total_g_loss.backward()

        if self.gradient_centralization:
            for param in chain(self.generators.parameters(), self.coders.parameters()):
                param.grad = self.centralized_grad(param)
        if self.adaptive_gradient_clipping:
            for param in chain(self.generators.parameters(), self.coders.parameters()):
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

    def write_tensorboard_summaries(self, batch, running, eval_running, running_steps, global_step):
        real_images, real_prev_images = batch
        real_images = torch.clone(real_images[:9])
        real_prev_images = torch.clone(real_prev_images[:9])
        # pictures and histograms
        with torch.no_grad():
            # Sample random points in the latent space
            batch_size = 9
            channels = self.code_features + self.noise_features
            if C_AXIS == 1:
                prev_size = [real_prev_images.shape[2],
                             real_prev_images.shape[3]]
                latent_shape = [batch_size, channels, *prev_size]
            else:
                prev_size = [real_prev_images.shape[1],
                             real_prev_images.shape[2]]
                latent_shape = [batch_size, *prev_size, channels]
            random_latent_vectors = torch.randn(
                size=latent_shape, device=self.device)
            coded_image = torch.cat(
                [real_prev_images, random_latent_vectors], dim=C_AXIS)
            # Decode them to fake images
            generated_images = self.generators(coded_image)
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
            for param in self.critics.parameters():
                with torch.no_grad():
                    critic_params = torch.cat(
                        [critic_params, torch.flatten(param.detach())])
            critic_params = critic_params.to(device=torch.device('cpu'))
            self.tensorboard_writer.add_histogram(
                'critics params', critic_params, global_step=global_step)

            coder_params = torch.tensor(
                [], dtype=torch.float32, device=self.device)
            for param in self.coders.parameters():
                with torch.no_grad():
                    coder_params = torch.cat(
                        [coder_params, torch.flatten(param.detach())])
            coder_params = coder_params.to(device=torch.device('cpu'))
            self.tensorboard_writer.add_histogram(
                'coders params', coder_params, global_step=global_step)

            generator_params = torch.tensor(
                [], dtype=torch.float32, device=self.device)
            for param in self.generators.parameters():
                with torch.no_grad():
                    generator_params = torch.cat(
                        [generator_params, torch.flatten(param.detach())])
            generator_params = generator_params.to(device=torch.device('cpu'))
            self.tensorboard_writer.add_histogram(
                'generators params', generator_params, global_step=global_step)

        # write scalars
        for scalar, value in running.items():
            self.tensorboard_writer.add_scalars(scalar,
                                                {"train": value / running_steps}, global_step)
        for scalar, value in eval_running.items():
            self.tensorboard_writer.add_scalars(scalar,
                                                {"test": value / running_steps}, global_step)
