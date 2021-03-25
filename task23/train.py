import argparse
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils import data

import dataloader as d
import model as m


def parse_args():
    """
    Create the argument parser for this learning task
    :return: parser object
    """
    parser = argparse.ArgumentParser(description="Script to train R2UNets and AttentionR2UNets on the cityscapes "
                                                 "dataset")
    parser.add_argument("-w", "--weights", dest="weights", type=str, default=None, nargs=1,
                        help="Filepath to the weights to use as a warm-start for the network, i.e. pretrained weights.")
    parser.add_argument("-l", "--weighted-loss", dest="weighted_loss", type=bool, default=False, action='store_true',
                        help="Flag to be set to use weighted cross_entropy loss for optimization.")
    parser.add_argument("-s", "--start", dest="start", type=int, nargs=1,
                        help="Start index of the training. Used for savinc the weigths to not override old results.")
    parser.add_argument("-e", "--end", dest="end", type=int, nargs=1, required=True, help="Number of epochs to train")
    parser.add_argument("-r", "--root", dest="root", type=str, nargs=1, required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("-t", "--target", dest="target", type=str, nargs=1, required=True,
                        help="Folder to store the weights in")
    parser.add_argument("--gpu", dest="gpu", type=int, default=-1, help="Set if GPU should be used if possible")
    parser.add_argument("-a", "--attention", dest="attention", type=bool, action='store_true', default=False,
                        help="Flag to be set to use attention in addition to R2U networks")
    return parser


if __name__ == '__main__':
    parser = parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
    results = parser.parse_args(sys.argv[1:])

    dataset_root_path = results.root + "data/cityscapes/"
    weights_path = results.target
    batch_size = 32
    num_classes = 19
    learning_rate = 0.001
    cpu_num = str(3)
    weights = [0.4795227666261817, 0.044115703166032035, 0.17762156254103204, 0.004551658309808298, 0.00604019549714417,
               0.002679068942029937, 0.0008393872685793067, 0.002815990287716649, 0.12196363945969013,
               0.007699043850938813, 0.0, 0.00708625601119354, 0.0009465834673713236, 0.0557585336380646,
               0.0023138812409729515, 0.0018495986040900736, 0.0018636213030133928, 0.0006389611508665966,
               0.002405177525111607, 0.07928837111016282]

    if torch.cuda.is_available() and results.gpu:
        dev = "cuda:" + cpu_num
        print("Using GPU")
    else:
        dev = "cpu"
        print("Using CPU")
    device = torch.device("cpu")

    if results.attention:
        model = m.AttR2UNet(num_classes + 1).to(device)
    else:
        model = m.R2UNet(num_classes + 1).to(device)

    if results.weights is not None:
        model.load(results.weights)
    print("Model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")

    train_set = d.cityscapesDataset(root=dataset_root_path, split="train")
    trainloader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    if results.weighted_loss:
        loss_f = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    episode_losses = []
    episode_accuracies = []

    for e in range(results.start, results.end):
        episode_loss, episode_accuracy = 0, 0

        for i, (image, label) in enumerate(trainloader):
            label = label.to(device)
            output = model.forward(image.to(device))
            loss = loss_f(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # do some statistics
            loss = loss.item()
            episode_loss += loss

            accuracy = torch.sum(output.argmax(dim=1).squeeze() == label).item() / (512 * 512 * batch_size)
            episode_accuracy += accuracy

            print("\rEpisode", e + 1, "/", results.end, "- Batch", i + 1, "/", len(train_set) // batch_size, "\tLoss:",
                  loss, "\tAcc:", accuracy, end="")

        '''
        update the general statistics
        Dividing by i is not 100% correct, because the last batch might be smaller
        but for evaluation issues, this can be ignored, because the general trend is
        shown
        '''
        episode_losses.append(episode_loss / i)
        episode_accuracies.append(episode_accuracy / i)
        print("\rEpisode", e + 1, "/", results.end, "- Completed \tLoss:", episode_losses[-1], "\tAcc:",
              episode_accuracies[-1])

        # save the models weights
        model.save(F"{weights_path}network_epoch{e}.pth")

    plt.title("Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.plot(episode_losses, "r")
    plt.savefig("./loss.png")
    plt.show()

    plt.title("Accuracy")
    plt.xlabel("Episodes")
    plt.ylabel("Accuracy")
    plt.plot(episode_accuracies, "b")
    plt.savefig("./accuracy.png")
    plt.show()
