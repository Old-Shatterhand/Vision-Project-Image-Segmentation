import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data

from task1.utils import pascalVOCDataset


def parse_args():
    """
    Create the argument parser for this learning task
    :return: parser object
    """
    parser = argparse.ArgumentParser(description="Script to count the number of pixels per class in the PASCAL VOC 2012"
                                                 "dataset.")
    parser.add_argument("-r", "--root", dest="root", type=str, nargs=1, required=True,
                        help="Root directory of the dataset. This should be the VOC_2012 folder of the dataset.")
    parser.add_argument("-t", "--transform", dest="transform", type=str, nargs=1, default=['lin'],
                        choices=['lin', 'log2', 'loge', 'log10'],
                        help="Transformation to perform on the final statistics before plotting")
    parser.add_argument("-o", "--output", dest="output", type=str, nargs=1, default=["./"],
                        help="Directory to store the file in. Will create a \"pascal_pixels.png\"-file.")
    parser.add_argument("-p", "--print", dest="print", default=False, action='store_true',
                        help="Flag indicating to print the exact number of pixels pwe classes.")
    return parser


def count_values(img, bins=21):
    counts = [0 for _ in range(bins)]
    for j in range(img.shape[0]):
        for i, t in enumerate(img[j, :].bincount(minlength=bins)):
            counts[i] += t.item()
    return counts


if __name__ == '__main__':
    # read the arguments
    parser = parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
    results = parser.parse_args(sys.argv[1:])

    # dataset variable
    # this step takes once around 15 minutes to create the masks
    dst = pascalVOCDataset(root=results.root[0], is_transform=True)

    # dataloader variable
    trainloader = data.DataLoader(dst, batch_size=1, shuffle=True)

    class_counts = [285742327, 3294319, 1407244, 3419625, 2461612, 2427801, 6644208, 5341379, 9872032, 4184171, 3184675,
                    4879887, 6162166, 3438894, 4167185, 16969029, 2576183, 3236067, 5255331, 5757808, 3356873]
    # class_counts = [0 for _ in range(21)]
    if class_counts[0] == 0:
        for j, (image, solution) in enumerate(trainloader):
            if (j + 1) % 10 == 0:
                print("\rImage", j + 1, "/", len(dst), end="")
            for i, c in enumerate(count_values(solution[0])):
                class_counts[i] += c

    if results.print:
        print(class_counts)

    class_counts = np.array(class_counts)
    if results.transform[0] == "log2":
        class_counts = np.log2(class_counts)
    elif results.transform[0] == "loge":
        class_counts = np.log(class_counts)
    elif results.transform[0] == "log10":
        class_counts = np.log10(class_counts)

    plt.bar(list(range(len(class_counts))), class_counts)
    plt.xlabel("Classes")
    plt.ylabel(F"Frequency ({results.transform[0]})")
    plt.title("Pixelwise Class Frequency in PASCAL VOC 2012 dataset")
    plt.savefig(os.path.join(results.output[0], "pascal_pixels.png"))
    plt.show()
