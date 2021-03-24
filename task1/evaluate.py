from utils import *


def image_auc(ground_truth, prediction):
    # prediction should be the classification
    prediction = prediction.view(21, 512 * 512).softmax(dim=0).transpose(0, 1).numpy()

    # this is done to ensure that every class occurs at least once in the true
    # labels. This is necessary, otherwise sklearn cannot compute the 
    # One-vs-all setting of the AUC. This will give not 100% correct results, 
    # but changing at most 21 of 262144 pixels (8*10^(-3)%) is not dramatic.
    for i in range(21):
        ground_truth[i] = i

    return roc_auc_score(ground_truth, prediction, average="weighted", multi_class="ovr", labels=list(range(21)))


def image_f1(ground_truth, prediction):
    return f1_score(ground_truth, prediction, average='weighted')


def image_dice(ground_truth, prediction):
    jaccard = jaccard_score(ground_truth, prediction, average='weighted')
    return (2 * jaccard) / (1 + jaccard)


def eval_epoch(epoch, length):
    # initialize the model
    # model = Segnet().to(device)
    model = Segnet(weights=F"{project_root_dir}/weights/network_epoch{epoch}.pth").to(device)
    f1, auc, dice, i = 0, 0, 0, 1

    for i, d in enumerate(valloader):
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


# define the root directories for the data
# project_root_dir = "C:/Users/joere/Google Drive/Project"
project_root_dir = "../datasets"
local_path = project_root_dir + '/data/VOCdevkit/VOC2012/'
# local_path = project_root_dir + '/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'

# define device to use
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device("cpu")

# initialize the dataset handlers
batch_size = 8
dst = pascalVOCDataset(root=local_path, is_transform=True)
valloader = data.DataLoader(dst, batch_size=batch_size, shuffle=False)
length = len(dst) / batch_size
f1_scores, aucs, dices = [], [], []
ofile = open("./metrics.txt", "w")

for e in range(0, 250):
    # f1_score, auc_score, dice_coefficient = eval_epoch(e)
    image_f1_score, image_auc_score, image_dice_coefficient = eval_epoch(e, length)
    f1_scores.append(image_f1_score)
    aucs.append(image_auc_score)
    dices.append(image_dice_coefficient)
    eval_string = F"Episode {e + 1} / 250 - F1: {image_f1_score} | AUC: {image_auc_score} | Dice: {image_dice_coefficient}"
    print(eval_string)
    ofile.write(eval_string)
    ofile.flush()

ofile.close()

plt.plot(f1_scores, label="F1")
plt.plot(aucs, label="AUC")
plt.plot(dices, label="Dice")
plt.show()
plt.savefig("./statistics.png")
