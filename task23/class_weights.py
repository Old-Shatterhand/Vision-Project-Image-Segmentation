import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data

import dataloader as d


def parse_args():
    """
    Create an argument parser for this script for better handling
    :return: Argument parser for the class weighting
    """
    parser = argparse.ArgumentParser(description="Script to count the number of pixels per class in the cityscapes"
                                                 "dataset.")
    parser.add_argument("-r", "--root", dest="root", type=str, nargs=1, required=True,
                        help="Root directory of the dataset. This should be the folder containing the \"leftImg8bit\" "
                             "folder and the \"gtFine\" folder of the dataset.")
    parser.add_argument("-t", "--transform", dest="transform", type=str, nargs=1, default=['lin'],
                        choices=['lin', 'log2', 'loge', 'log10'],
                        help="Transformation to perform on the final statistics before plotting")
    parser.add_argument("-o", "--output", dest="output", type=str, nargs=1, default=["./"],
                        help="Directory to store the file in. This will create a \"cityscapes_pixels.png\"-file.")
    parser.add_argument("-p", "--print", default=False, action='store_true',
                        help="Flag indicating to print the exact number of pixels pwe classes.")
    return parser


def check_arg_validity(args):
    """
    Check arguments for validity and report invalid arguments
    :param args: parsed arguments
    :return: True if all arguments are valid, False otherwise
    """
    result = True

    # check the root directory
    if not os.path.isdir(args.root[0]):
        result = False
        print("Root directory does not exist.")
    else:
        # and if the expected children are contained
        sub_dirs = [os.path.join(args.root[0], o) for o in os.listdir(args.root[0])
                    if os.path.isdir(os.path.join(args.root[0], o))]
        if "gtFine" not in sub_dirs or "leftImg8bit" not in sub_dirs:
            result = False
            print("Database root has not the expected subdirectories.")

    # check the output directory
    if not os.path.isdir(args.output[0]):
        result = False
        print("Output directory does not exist.")

    return result


def count_values(img, bins=20):
    """
    count the pixelwise occurrences in each class
    :param img: input image
    :param bins: number of different classes
    :return: pixel counts per class
    """
    counts = [0 for _ in range(bins)]
    for j in range(img.shape[0]):
        for i, t in enumerate(img[j, :].bincount(minlength=bins)):
            counts[i] += t.item()
    return counts


if __name__ == '__main__':
    # read the arguments
    parser = parse_args()
    results = parser.parse_args(sys.argv[1:])
    if not check_arg_validity(results):
        exit(1)

    # initialize some fields for the statistics
    num_classes = 20
    total_class_counts = [0 for _ in range(num_classes)]

    # Because this is a challenge, the labeling of the testset is not provided or just black
    for i, s in enumerate(["train", "val"]):
        dataset = d.cityscapesDataset(root=results.root[0], split=s)
        loader = data.DataLoader(dataset, batch_size=1)
        class_counts = [0 for _ in range(num_classes)]

        for j, (image, solution) in enumerate(loader):
            if (j + 1) % 10 == 0:
                print("\rSplit", s, "- Image", j + 1, "/", len(dataset), end="")

            for k, c in enumerate(count_values(solution[0])):
                class_counts[k] += c
                total_class_counts[k] += c

        print()
        if results.print:
            print(s + ":")
            print(class_counts)

    if results.print:
        print("Total:")
        print(total_class_counts)

    # transform the counts according to the provided transformation
    total_class_counts = np.array(total_class_counts)
    if results.transform[0] == "log2":
        total_class_counts = np.log2(total_class_counts)
    elif results.transform[0] == "loge":
        total_class_counts = np.log(total_class_counts)
    elif results.transform[0] == "log10":
        total_class_counts = np.log10(total_class_counts)

    # plot the transformed counts as a bar chart
    plt.bar(list(range(len(total_class_counts))), total_class_counts)
    plt.xlabel("Classes")
    plt.ylabel(F"Frequency ({results.transform[0]})")
    plt.title("Pixelwise Class Frequency in cityscapes dataset")
    plt.savefig(os.path.join(results.output[0], "cityscapes_pixels.png"))
    plt.show()
