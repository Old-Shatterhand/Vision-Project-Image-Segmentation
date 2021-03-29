import model as m
import dataloader as d

import torch
from torch.utils import data
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse
import sys
import os


def parse_args():
    """
    Create an argument parser for this script for better handling
    :return: Argument parser for the visualization
    """
    parser = argparse.ArgumentParser(description="Visualize classifications from the trained models.")
    parser.add_argument("--r2u", dest="r2u", type=str, required=True, nargs=1,
                        help="Path to the weights file of the R2UNet to use for classification.")
    parser.add_argument("--w-r2u", dest="w_r2u", type=str, required=True, nargs=1,
                        help="Path to the weights file of the R2UNet trained with weighted CE-Loss to use for "
                             "classification.")
    parser.add_argument("--attr2u", dest="att_r2u", type=str, required=True, nargs=1,
                        help="Path to the weights file of the AttentionR2UNet to use for classification.")
    parser.add_argument("--w-attr2u", dest="w_att_r2u", type=str, required=True, nargs=1,
                        help="Path to the weights file of the AttentionR2UNet trained with weighted CE-Loss to use for "
                             "classification.")
    parser.add_argument("--gpu", dest="gpu", type=int, default=-1,
                        help="Set number of if GPU should be used if possible")
    parser.add_argument("-r", "--root", dest="root", type=str, nargs=1, required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("-o", "--output", dest="output", type=str, nargs=1, default=["./"],
                        help="Directory to store the file in. This will create a \"r2u_comparison.png\"-file.")
    return parser


if __name__ == '__main__':
    # Read the arguments
    parser = parse_args()
    results = parser.parse_args(sys.argv[1:])

    # set the device to use for computation, i.e. run on CPU or GPU
    if torch.cuda.is_available() and results.gpu != -1:
        dev = "cuda:" + results.gpu
        print("Using GPU")
    else:
        dev = "cpu"
        print("Using CPU")
    device = torch.device(dev)

    # initialize the four models to use for visualization
    models = [
        m.R2UNet(20, weights=results.r2u[0]), m.R2UNet(20, weights=results.w_r2u[0]),
        m.AttR2UNet(20, weights=results.att_r2u[0]), m.AttR2UNet(20, weights=results.w_att_r2u[0])
    ]

    # initialize other variables necessary for visualization
    torch.manual_seed(6467)
    dst = d.cityscapesDataset(root=results.root[0])
    val_loader = data.DataLoader(dst, batch_size=1, shuffle=True)

    # extract an image and the according solution
    image, solution = next(iter(val_loader))

    # compute the classifications from the models
    model_outputs = []
    with torch.no_grad():
        for i in range(4):
            model_outputs.append(models[i].classify(image.to(device)).squeeze().cpu())

    # initialize the colormaps to use for plotting the classifications
    cmap = colors.ListedColormap(dst.get_cs_labels() / 255)
    bounds = list(range(20))
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(12, 9))

    # add the original image
    im1 = fig.add_subplot(321)
    im1.imshow(image[0].transpose(0, 2).transpose(0, 1).numpy())
    im1.axis("off")

    # add the solution
    im2 = fig.add_subplot(322)
    im2.imshow(solution[0].numpy().astype('uint8'), cmap=cmap, norm=norm)
    im2.axis("off")

    # add the classification of the basic R2UNet
    im3 = fig.add_subplot(323)
    im3.imshow(model_outputs[0].numpy().astype('uint8'), cmap=cmap, norm=norm)
    im3.axis("off")

    # add the classification of the basic R2UNet trained with weighted CE-loss
    im4 = fig.add_subplot(324)
    im4.imshow(model_outputs[1].numpy().astype('uint8'), cmap=cmap, norm=norm)
    im4.axis("off")

    # add the classification of the AttentionR2UNet
    im5 = fig.add_subplot(325)
    im5.imshow(model_outputs[2].numpy().astype('uint8'), cmap=cmap, norm=norm)
    im5.axis("off")

    # add the classification of the AttentionR2UNet with weighted CE-Loss
    im6 = fig.add_subplot(326)
    im6.imshow(model_outputs[3].numpy().astype('uint8'), cmap=cmap, norm=norm)
    im6.axis("off")

    # save and plot the resulting image
    fig.tight_layout()
    plt.savefig(os.path.join(results.output[0], "r2u_comparison.png"))
    plt.show()
