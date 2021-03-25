from utils import *
import sys


def image_auc(ground_truth, prediction):
    """
    Compute the AUC of the input image and the ground truth
    :param ground_truth: true labels of each pixel
    :param prediction: predictions for each classes and pixel, i.e. the
    :return: AUC of the input image compared to the ground truth
    """
    # prediction should be the classification, not the argmaxed network-output
    prediction = prediction.view(21, 512 * 512).softmax(dim=0).transpose(0, 1).numpy()

    # this is done to ensure that every class occurs at least once in the true
    # labels. This is necessary, otherwise sklearn cannot compute the 
    # One-vs-all setting of the AUC. This will give not 100% correct results, 
    # but changing at most 21 of 262144 pixels (8*10^(-3)%) is not dramatic.
    for i in range(21):
        ground_truth[i] = i

    return roc_auc_score(ground_truth, prediction, average="weighted", multi_class="ovr", labels=list(range(21)))


def image_f1(ground_truth, prediction):
    """
    Compute the F1 score of the input image and the ground truth
    :param ground_truth: true labels for each pixel
    :param prediction: predicted classes for each pixel
    :return: f1 score of the input image compared to the ground truth
    """
    return f1_score(ground_truth, prediction, average='weighted')


def image_dice(ground_truth, prediction):
    """
    Compute the dice_coefficient based on the jaccard score as they are convertible using the formula provided on
    wikipedia: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient#Difference_from_Jaccard
    :param ground_truth: true labels of each pixel
    :param prediction: predicted classes for each pixel
    :return: soerensen-dice-coefficient of the input image compared to the ground truth
    """
    jaccard = jaccard_score(ground_truth, prediction, average='weighted')
    return (2 * jaccard) / (1 + jaccard)


def eval_epoch(epoch, length, valloader):
    """
    Evaluate an epoch
    :param epoch:
    :param length:
    :param valloader:
    :return:
    """
    # initialize the model
    model = Segnet(weights=F"{project_root_dir}/weights/network_epoch{epoch}.pth").to(device)

    f1, auc, dice, i = 0, 0, 0, 1
    for i, d in enumerate(valloader):
        # print the progress
        print("\r", i, "/", length, end="")
        with torch.no_grad():
            prediction = model.forward(d[0].to(device)).cpu().squeeze()
            label_array = d[1].view(batch_size, 512 * 512).numpy()
            prediction_array = prediction.argmax(dim=1).squeeze().view(batch_size, 512 * 512).numpy()

            for b in range(batch_size):
                f1 += image_f1(label_array[b], prediction_array[b])
                auc += image_auc(label_array[b], prediction[b])
                dice += image_dice(label_array[b], prediction_array[b])

    return f1 / i, auc / i, dice / i


if __name__ == '__main__':
    # define the root directories for the data
    project_root_dir = sys.argv[1]  # "../datasets"
    local_path = project_root_dir + '/data/VOCdevkit/VOC2012/'

    # initialize the dataset, dataloader, and other utils for the evaluation of task 1
    device = torch.device("cpu")
    batch_size = 8
    dst = pascalVOCDataset(root=local_path, is_transform=True)
    valloader = data.DataLoader(dst, batch_size=batch_size, shuffle=False)
    length = len(dst) / batch_size
    f1_scores, aucs, dices = [], [], []
    ofile = open("./metrics.txt", "w")

    # iterate over all episodes and evaluate the network
    for e in range(0, 250):
        image_f1_score, image_auc_score, image_dice_coefficient = eval_epoch(e, length, valloader)

        f1_scores.append(image_f1_score)
        aucs.append(image_auc_score)
        dices.append(image_dice_coefficient)

        eval_string = F"Episode {e + 1} / 250 - F1: {image_f1_score} | AUC: {image_auc_score} | Dice: {image_dice_coefficient}"
        print(eval_string)

        ofile.write(eval_string)
        ofile.flush()

    ofile.close()

    # save a figure with the three lines from the evaluation
    plt.plot(f1_scores, label="F1")
    plt.plot(aucs, label="AUC")
    plt.plot(dices, label="Dice")
    plt.legend(loc="lower right")
    plt.savefig("./statistics.png")
    plt.show()
