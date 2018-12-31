from __future__ import print_function
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
from PIL import Image

import models.dcgan as dcgan
import models.mlp as mlp


class ImageWizard():

    def __init__(self, cfg_path, weights_path):

        with open(cfg_path, 'r') as gencfg:
            generator_config = json.loads(gencfg.read())

        imageSize = generator_config["imageSize"]
        nz = generator_config["nz"]
        nc = generator_config["nc"]
        ngf = generator_config["ngf"]
        noBN = generator_config["noBN"]
        ngpu = generator_config["ngpu"]
        mlp_G = generator_config["mlp_G"]
        n_extra_layers = generator_config["n_extra_layers"]

        if noBN:
            netG = dcgan.DCGAN_G_nobn(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
        elif mlp_G:
            netG = mlp.MLP_G(imageSize, nz, nc, ngf, ngpu)
        else:
            netG = dcgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu, n_extra_layers)

        # load weights
        netG.load_state_dict(torch.load(weights_path))

        if torch.cuda.is_available():
            netG = netG.cuda()

        self.model = netG
        self.nz = nz

    
    def generate_palette(self, nimages=100, dim_z=None):

        z = np.random.randn(1, self.nz, 1, 1)
        Z = np.tile(z, (nimages, 1, 1, 1))
        Z = Z.astype(np.float32)

        if dim_z is None:
            dim_z = np.random.choice(self.nz)

        # bugfix
        Z[:, dim_z, 0, 0] = np.linspace(-1, 1, nimages)

        Z = torch.tensor(Z)

        if torch.cuda.is_available():
            Z = Z.cuda()

        # forward pass
        G = self.model(Z)
        G.data = G.data.mul(0.5).add(0.5)

        images_list = list()
        for i in range(nimages):
            ndarr = G.data[i, ...].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(ndarr)
            images_list.append(im)

        return images_list



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='path to generator config .json file')
    parser.add_argument('-w', '--weights', required=True, type=str, help='path to generator weights .pth file')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help="path to to output directory")
    parser.add_argument('-n', '--nimages', required=True, type=int, help="number of images to generate", default=1)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    opt = parser.parse_args()

    iw = ImageWizard(opt.config, opt.weights)

    img_arr = iw.generate_palette(opt.nimages)

    for i, img in enumerate(img_arr):
        img.save(os.path.join(opt.output_dir, "%03d.png"%i))
