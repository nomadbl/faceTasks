import subprocess
import argparse
from unet_segmentor_lib import *
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



def main_gan(station):
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
    :return:
    """

    home = None
    if station == "aws":
        home = os.getcwd()
    elif station == "home":
        home = "/home/lior/PycharmProjects/facesTasks"

    print("current working directory:", home)
    files_dir = f'{home}/images/*.jpg'
    cp_dir = f'{home}/session/pt_checkpoint/'
    latent_features = 215
    code_features = 16
    noise_features = latent_features - code_features
    gan = InfoWGAN(files_dir=files_dir,
                   code_features=code_features, noise_features=noise_features,
                   pixel_features=8,
                   decoder_filters_list=[64, 64, 32, 32, 16, 16, 8],
                   cp_dir=cp_dir,
                   epochs_per_phase=1,
                   info_lambda=100,
                   grad_lambda=10)
    gan.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", dest="v", type=int, choices={1, 2}, default=2, required=False, help="1 - unet, 2 - gan")
    parser.add_argument("--station", dest="station", type=str, choices={"aws", "home"}, default="aws", required=False)
    args = parser.parse_args()

    print(f"running pytorch version {torch.__version__}")
    if args.station == "aws":
        subprocess.run(["nvidia-smi", "-L"])

    # if args.v == 1:
    #     main_unet(args.station)

    if args.v == 2:
        main_gan(args.station)
