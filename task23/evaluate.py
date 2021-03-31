import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import jaccard_score, multilabel_confusion_matrix, f1_score
from torch.utils import data

import dataloader as d
import model as m


def parse_args():
    """
    Create an argument parser for this script for better handling
    :return: Argument parser for the evaluation
    """
    parser = argparse.ArgumentParser(description="Script to evaluate R2UNets and AttentionR2UNets on the cityscapes "
                                                 "dataset")
    parser.add_argument("-w", "--weights", dest="weights", type=str, required=True, nargs=1,
                        help="Directory of the weights to use for the evaluation of the network, "
                             "i.e. pretrained weights.")
    parser.add_argument("-s", "--start", dest="start", type=int, nargs=1, default=[0],
                        help="Start index of the evaluation.")
    parser.add_argument("-e", "--end", dest="end", type=int, nargs=1, required=True,
                        help="Number of epochs that were trained and should now be evaluated")
    parser.add_argument("-r", "--root", dest="root", type=str, nargs=1, required=True,
                        help="Root directory of the dataset. This should be the folder containing the \"leftImg8bit\" "
                             "folder and the \"gtFine\" folder of the dataset.")
    parser.add_argument("--gpu", dest="gpu", type=int, default=[-1],
                        help="Set number of if GPU should be used if possible")
    parser.add_argument("-a", "--attention", dest="attention", action='store_true', default=False,
                        help="Flag to be set to use attention in addition to R2U networks")
    parser.add_argument("-o", "--output", dest="output", type=str, nargs=1, default=["./"],
                        help="Directory to store the files in. This will create a \"metrics.txt\"-file and a "
                             "\"statistics.png\"-file.")
    parser.add_argument("-c", "--class", dest="classes", default=False, action='store_true',
                        help="Flag indicating to do class-wise evaluation, i.e. report the metrics class-wise. This "
                             "evaluation will only be performed for the last (end-1-th) episode.")
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

    # check the directory of the weights
    if args.weights is not None and not os.path.exists(args.weights[0]):
        result = False
        print("Weights file does not exist.")

    return result


def image_jaccard(ground_truth, prediction, classes):
    """
    Compute the Jaccard index of the predicted classes related to the ground truth image
    :param ground_truth: true labels for each pixel
    :param prediction: predicted classes for each pixel
    :return: Jaccard score of the input image compare to the ground truth
    """
    if classes:
        return jaccard_score(ground_truth, prediction, average=None, labels=list(range(20)))
    return jaccard_score(ground_truth, prediction, average='weighted')


def image_f1(ground_truth, prediction, classes):
    """
    Compute the F1 score of the input image and the ground truth
    :param ground_truth: true labels for each pixel
    :param prediction: predicted classes for each pixel
    :return: f1 score of the input image compared to the ground truth
    """
    if classes:
        return f1_score(ground_truth, prediction, average=None, labels=list(range(20)))
    return f1_score(ground_truth, prediction, average='weighted')


def image_dice(ground_truth, prediction, classes):
    """
    Compute the dice_coefficient based on the jaccard score as they are convertible using the formula provided on
    wikipedia: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient#Difference_from_Jaccard
    :param ground_truth: true labels of each pixel
    :param prediction: predicted classes for each pixel
    :return: soerensen-dice-coefficient of the input image compared to the ground truth
    """
    if classes:
        jaccard = jaccard_score(ground_truth, prediction, average=None, labels=list(range(20)))
    else:
        jaccard = jaccard_score(ground_truth, prediction, average='weighted')
    return (2 * jaccard) / (1 + jaccard)


def image_sensitivity_specificity(ground_truth, prediction, weights):
    """
    Compute sensitivity and specificity for each class and a weighted mean
    :param ground_truth: true labels of each pixel
    :param prediction: predicted classes for each pixel
    :param weights: weights to use for averaging the classes
    :return: mean sensitivity, mean specificity
    """
    confusion_matrix = multilabel_confusion_matrix(ground_truth, prediction, labels=list(range(20)))
    class_sensitivity = np.nan_to_num(np.divide(confusion_matrix[:, 1, 1],
                                                (confusion_matrix[:, 1, 1] + confusion_matrix[:, 1, 0])),
                                      neginf=0, posinf=0)
    class_specificity = np.nan_to_num(np.divide(confusion_matrix[:, 0, 0],
                                                (confusion_matrix[:, 0, 0] + confusion_matrix[:, 0, 1])),
                                      neginf=0, posinf=0)
    sensitivity = np.sum(class_sensitivity * weights)
    specificity = np.sum(class_specificity * weights)
    return sensitivity, specificity, class_sensitivity, class_specificity


def eval_epoch(epoch, length, val_loader, weights_dir, model_class, device, weights, batch_size, classes):
    """
    Evaluate an epoch of task 2 and 3
    :param epoch: number of the epoch to evaluate
    :param length: number of images in the dataset
    :param val_loader: dataloader to load the validation images from
    :param weights_dir: directory with the weights
    :return: performance measures of this epoch
    """
    # initialize the model
    model = model_class(num_classes=20, weights=os.path.join(weights_dir, F"network_epoch{epoch}.pth")).to(device)

    # initialize the epoch measures
    sensitivity, specificity, class_sensitivity, class_specificity, jaccard, f1, dice, i = \
        0, 0, np.zeros(20), np.zeros(20), 0, 0, 0, 1

    # iterate over all batched
    for i, d in enumerate(val_loader):
        # print the progress
        print("\r", i, "/", length / batch_size, end="")

        # for simple forwarding not gradients need to be computed
        with torch.no_grad():
            # compute the classification of the images
            prediction = model.forward(d[0].to(device)).cpu().squeeze()
            label_array = d[1].view(batch_size, 512 * 256).numpy()
            prediction_array = prediction.argmax(dim=1).squeeze().view(batch_size, 512 * 256).numpy()

            # for each image of the batch compute the performance measures
            for ground_truth, prediction in zip(label_array, prediction_array):
                sense, spec, class_sense, class_spec = image_sensitivity_specificity(ground_truth, prediction, weights)

                # add to epoch specificity and sensitivity
                sensitivity += sense
                specificity += spec
                class_sensitivity += class_sense
                class_specificity += class_spec

                # add to other epoch-wise measures
                jaccard += image_jaccard(ground_truth, prediction, classes)
                f1 += image_f1(ground_truth, prediction, classes)
                dice += image_dice(ground_truth, prediction, classes)

    # return the averaged performance measures
    return sensitivity / length, specificity / length, class_sensitivity / length, class_specificity / length, \
           jaccard / length, f1 / length, dice / length


if __name__ == '__main__':
    # read the arguments
    parser = parse_args()
    results = parser.parse_args(sys.argv[1:])

    # the weights are computes as the fraction of each class in the total number of classes per pixel
    weights = [0.4795227666261817, 0.044115703166032035, 0.17762156254103204, 0.004551658309808298, 0.00604019549714417,
               0.002679068942029937, 0.0008393872685793067, 0.002815990287716649, 0.12196363945969013,
               0.007699043850938813, 0.0, 0.00708625601119354, 0.0009465834673713236, 0.0557585336380646,
               0.0023138812409729515, 0.0018495986040900736, 0.0018636213030133928, 0.0006389611508665966,
               0.002405177525111607, 0.07928837111016282]

    # set the device to use for computation, i.e. run on CPU or GPU
    if torch.cuda.is_available() and results.gpu[0] != -1:
        dev = "cuda:" + results.gpu[0]
        print("Using GPU")
    else:
        dev = "cpu"
        print("Using CPU")
    device = torch.device("cpu")

    # initialize the dataset and the according dataloader
    batch_size = 10
    val_set = d.cityscapesDataset(root=results.root[0], split="val")
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # create an output file for detailed measures
    ofile = open(os.path.join(results.output[0], "metrics.txt", "w"))

    # initialize the model
    model = m.AttR2UNet if results.attention else m.R2UNet

    # initialize the performance measures
    sensitivities, specificities, f1_scores, jaccard_scores, dice_scores = [], [], [], [], []

    if results.classes:
        # perform class-wise evaluation of the last epoch
        sensitivity, specificity, class_sensitivity, class_specificity, jaccard, f1, dice = \
            eval_epoch(results.end[0] - 1, len(val_set), val_loader, results.weights[0], model, device, weights,
                       batch_size, True)

        # print the results
        print(F"Episode {results.end[0]} - Class-wise Evaluation:")
        for i, (se, sp, f, j, d) in enumerate(zip(class_sensitivity, class_specificity, f1, jaccard, dice)):
            print(val_set.get_cs_labels()[i], ": SE:", se, "- SP:", sp, "- F1:", f, "- JA:", j, "- SD:", d)

    else:
        # iterate over all episodes and evaluate the network
        for e in range(results.start[0], results.end[0]):
            sensitivity, specificity, class_sensitivity, class_specificity, jaccard, f1, dice = \
                eval_epoch(e, len(val_set), val_loader, results.weights[0], model, device, weights, batch_size, False)

            # update the statistics
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            f1_scores.append(f1)
            jaccard_scores.append(jaccard)
            dice_scores.append(dice)

            # print the actual performance of episode e
            eval_string = F"Episode {e + 1} / {results.end[0] - results.start[0]} - Se: {sensitivity} | " \
                          F"Sp: {specificity} | F1: {f1} | Jaccard: {jaccard} | Dice: {dice}"
            print(eval_string)

            # save the data to the output file
            ofile.write(eval_string)
            ofile.write("\t" + str(class_sensitivity))
            ofile.write("\t" + str(class_specificity))
            ofile.flush()

        ofile.close()

        # save a figure with the three lines from the evaluation
        plt.plot(sensitivities, label="Sens.")
        plt.plot(specificities, label="Spec.")
        plt.plot(f1_scores, label="F1")
        plt.plot(jaccard_scores, label="Jaccard")
        plt.plot(dice_scores, label="Dice")
        plt.legend(loc="lower right")
        plt.title("Evaluation")
        plt.xlabel("Episodes")
        plt.savefig(os.path.join(results.output[0], "statistics.png"))
        plt.show()
