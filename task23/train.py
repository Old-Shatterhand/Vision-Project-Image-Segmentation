import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils import data

import dataloader as d
import model as m


def parse_args():
    """
    Create an argument parser for this script for better handling
    :return: Argument parser for the training
    """
    parser = argparse.ArgumentParser(description="Script to train R2UNets and AttentionR2UNets on the cityscapes "
                                                 "dataset")
    parser.add_argument("-w", "--weights", dest="weights", type=str, default=None, nargs=1,
                        help="Filepath to the weights to use as a warm-start for the network, i.e. pretrained weights.")
    parser.add_argument("-l", "--weighted-loss", dest="weighted_loss", default=False, action='store_true',
                        help="Flag to be set to use weighted cross_entropy loss for optimization.")
    parser.add_argument("-s", "--start", dest="start", type=int, nargs=1, default=0,
                        help="Start index of the training. Used for saving the weights to not override old results.")
    parser.add_argument("-e", "--end", dest="end", type=int, nargs=1, required=True, help="Number of epochs to train.")
    parser.add_argument("-r", "--root", dest="root", type=str, nargs=1, required=True,
                        help="Root directory of the dataset. This folder should contain the \"gtFine\" folder and the "
                             "\"leftImg8bit\" folder.")
    parser.add_argument("-t", "--target", dest="target", type=str, nargs=1, required=True,
                        help="Folder to store the weights in.")
    parser.add_argument("--gpu", dest="gpu", type=int, default=[-1],
                        help="Set number of if GPU should be used if possible")
    parser.add_argument("-a", "--attention", dest="attention", action='store_true', default=False,
                        help="Flag to be set to use attention in addition to R2U networks")
    parser.add_argument("-o", "--output", dest="output", default=["./"], type=str, nargs=1,
                        help="Directory to store the curves in. This will create a \"loss.png\"-file and "
                             "\"accuracy\"-file.")
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

    # check the target directory
    if not os.path.isdir(args.target[0]):
        result = False
        print("Target directory does not exist.")

    # check the directory of the weights
    if args.weights is not None and not os.path.exists(args.weights[0]):
        result = False
        print("Weights file does not exist.")

    return result


if __name__ == '__main__':
    # read the arguments
    parser = parse_args()
    results = parser.parse_args(sys.argv[1:])

    # set the device to use for computation, i.e. run on CPU or GPU
    if torch.cuda.is_available() and results.gpu[0] != -1:
        dev = "cuda:" + results.gpu[0]
        print("Using GPU")
    else:
        dev = "cpu"
        print("Using CPU")
    device = torch.device(dev)

    # initialize some hyperparameter of the training
    batch_size = 32
    num_classes = 19
    learning_rate = 0.001
    weights = [0.4795227666261817, 0.044115703166032035, 0.17762156254103204, 0.004551658309808298, 0.00604019549714417,
               0.002679068942029937, 0.0008393872685793067, 0.002815990287716649, 0.12196363945969013,
               0.007699043850938813, 0.0, 0.00708625601119354, 0.0009465834673713236, 0.0557585336380646,
               0.0023138812409729515, 0.0018495986040900736, 0.0018636213030133928, 0.0006389611508665966,
               0.002405177525111607, 0.07928837111016282]

    # initialize the network to use
    if results.attention:
        model = m.AttR2UNet(num_classes + 1).to(device)
    else:
        model = m.R2UNet(num_classes + 1).to(device)

    # load the weights to the network if some are provided
    if results.weights is not None:
        model.load(results.weights[0])
    print("Model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")

    # initialize te dataset and the dataloader
    train_set = d.cityscapesDataset(root=results.root[0], split="train")
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # initialize the loss-function, if necessary with weights, and the optimizer
    if results.weighted_loss:
        loss_f = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
    else:
        loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # initialize some statistics
    episode_losses = []
    episode_accuracies = []

    # start the training loop
    for e in range(results.start[0], results.end[0]):
        # some episode statistics
        episode_loss, episode_accuracy, i = 0, 0, 1

        # iterate over all batched in the dataloader
        for i, (image, label) in enumerate(train_loader):
            # forward the image through the network and compute the loss
            label = label.to(device)
            output = model.forward(image.to(device))
            loss = loss_f(output, label)

            # perform backpropagation through the network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # do some statistics
            loss = loss.item()
            episode_loss += loss

            accuracy = torch.sum(output.argmax(dim=1).squeeze() == label).item() / (512 * 256 * batch_size)
            episode_accuracy += accuracy

            print("\rEpisode", e + 1, "/", results.end[0], "- Batch", i + 1, "/", len(train_set) // batch_size,
                  "\tLoss:", loss, "\tAcc:", accuracy, end="")

        '''
        update the general statistics
        Dividing by i is not 100% correct, because the last batch might be smaller
        but for evaluation issues, this can be ignored, because the general trend is
        shown
        '''
        episode_losses.append(episode_loss / i)
        episode_accuracies.append(episode_accuracy / i)
        print("\rEpisode", e + 1, "/", results.end[0], "- Completed \tLoss:", episode_losses[-1], "\tAcc:",
              episode_accuracies[-1])

        # save the models weights
        model.save(os.path.join(results.target[0], F"network_epoch{e}.pth"))

    # plot the loss-curve of the training
    plt.title("Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.plot(episode_losses, "r")
    plt.savefig(os.path.join(results.output[0], "loss.png"))
    plt.show()

    # plot the accuracy-curve of the training
    plt.title("Accuracy")
    plt.xlabel("Episodes")
    plt.ylabel("Accuracy")
    plt.plot(episode_accuracies, "b")
    plt.savefig(os.path.join(results.output[0], "accuracy.png"))
    plt.show()
