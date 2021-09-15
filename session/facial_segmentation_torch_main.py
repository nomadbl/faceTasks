import os
import subprocess
import argparse
from pgan_torch import InfoWGAN
import torch


def main_unet(station):
    # station = "home"
    # station = "aws"
    train = False
    evaluate = False

    home = None
    if station == "aws":
        home = os.getcwd()
    elif station == "home":
        home = "/home/lior/PycharmProjects/facesTasks"

    print("current working directory:", home)

    # AE V1
    # bridge_features = [16]
    # encoder_filters_list = [ 64, 64, 128, 128, 256, 256]
    # decoder_filters_list = [256, 256, 128, 128, 64, 64]
    # head_filters_list = [32, 32, 16, 3]
    bridge_features = [16, 16, 16, 16]
    encoder_filters_list = [128, 64, 32, 32, 32]
    skip_connections = [1, 1, 1, 0, 0]
    decoder_filters_list = [32, 32, 32, 64, 128]
    head_filters_list = [3]


def main_gan(station, max_image_shape, cp_dir=None, lr=0.001, adaptive_gradient_clipping=False, gradient_centralization=False, next=False):
    """
    Attempt #2
    We will use a GAN architecture similar to infoGAN: https://arxiv.org/pdf/1606.03657.pdf
    However there are several changes relative to this work:
    1. As we want a segmentation map, we need to generate a code per pixel.
    Thus the architecture will be similar to what we tried above with two changes.
    We will not use skip connections between the encoder and decoder so we can clearly identify the code.
    Also we will use "same" padding and strided convolutions to mimic the downsampling while generating a code
    for each pixel.

    2. As was shown just above, maximizing the mutual information is the same as minimizing the distance between
    the decoded signal and the original signal. In infoGAN the mutual information I(c;G(c,z)) is maximized.
    Therefore we will minimize the distance L=|c-E(G(c,z))|^2. That is, we use the encoder part on the generated
    image to generate a code as similar as possible to the original one that was sampled while training the GAN.
    This part can be trained seperately from the generator (i.e. decoder) and the discriminator.
    Therefore the segmentor is actually the encoder which we will get "for free" with this method.
    3. I will use the WGAN method of https://arxiv.org/pdf/1701.07875.pdf.
    Additionaly we use the gradient penalty of https://arxiv.org/pdf/1704.00028.pdf

    To reduce training time the discriminator can actually be trained as an alternate "head" of the encoder.
    This is actually what was done in the infoGAN paper.
    In drawing samples from the latent space, we use a gaussian of unit variance.
    Since each pixel has N components in the code, we need each component to be a gaussian of variance 1/sqrt(N)
    (this way the total variance is 1).
    :param station: {"aws", "home"}. run locally or on cloud
    :param cp_dir
    :param max_image_shape: int. X (image_shape[0]) resolution of images at desired output
    :return:
    """
    if type(cp_dir) is str and '~' in cp_dir:
        homedir = os.getenv('HOME')
        cp_dir = cp_dir.replace('~', homedir)
    home = None
    if station == "aws":
        home = os.getcwd()
    elif station == "home":
        home = "/home/lior/PycharmProjects/facesTasks"

    print("current working directory:", home)
    files_dir = f'{home}/images/*.jpg'
    i = 0
    if cp_dir is None:
        while os.path.exists(f'{home}/session/run_{i+1}/'):
            i += 1
        os.mkdir(f'{home}/session/run_{i+1}/')
        cp_dir = f'{home}/session/run_{i + 1}/'
    else:
        if not os.path.exists(cp_dir):
            raise ValueError("Checkpoint dir does not exist")
    latent_features = 215
    code_features = 16
    noise_features = latent_features - code_features
    gan = InfoWGAN(files_dir=files_dir,
                   code_features=code_features, noise_features=noise_features,
                   pixel_features=8,
                   decoder_filters_list=[64, 64, 64, 32, 16, 16, 8],
                   cp_dir=cp_dir,
                   epochs_per_phase=100,
                   info_lambda=100,
                   grad_lambda=10,
                   lr=lr,
                   adaptive_gradient_clipping=adaptive_gradient_clipping,
                   gradient_centralization=gradient_centralization,
                   max_image_shape=max_image_shape,
                   start_from_next_resolution=next)
    gan.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", dest="v", type=int,
                        choices={1, 2}, default=2, required=False, help="1 - unet, 2 - gan")
    parser.add_argument("--station", dest="station", type=str,
                        choices={"aws", "home"}, default="aws", required=False, help="local machine or aws deployment")
    parser.add_argument("--checkpoint", "--c",
                        dest="cp_dir", type=str, required=False, help="resume training from latest checkpoint in specified folder")
    parser.add_argument("--lr", type=float, dest="lr",
                        required=False, default=0.001, help="learning rate")
    parser.add_argument("--agc", action="store_true",
                        help="activate adaptive gradient clipping")
    parser.add_argument("--gc", action="store_true",
                        help="activate gradient centering algorithm")
    parser.add_argument("--next", action="store_true",
                        help="skip to next image resolution in pGAN training")
    args = parser.parse_args()

    print(f"running pytorch version {torch.__version__}")
    if args.station == "aws":
        subprocess.run(["nvidia-smi", "-L"])

    # if args.v == 1:
    #     main_unet(args.station)

    if args.v == 2:
        main_gan(args.station, cp_dir=args.cp_dir,
                 max_image_shape=128, lr=args.lr, adaptive_gradient_clipping=args.agc, gradient_centralization=args.gc, next=args.next)
