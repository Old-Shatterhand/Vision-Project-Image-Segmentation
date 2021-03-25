import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils import data

import dataloader as d
import model as m

project_root_dir = "../../datasets/"
dataset_root_path = project_root_dir + "data/cityscapes/"
weights_path = project_root_dir + "weights/task2/"
batch_size = 32
start_epoch = 0
end_epoch = 100
num_classes = 19
learning_rate = 0.001
weighted_loss = False
cpu_num = str(3)

if torch.cuda.is_available():
    dev = "cuda:" + cpu_num
    print("Using GPU")
else:
    dev = "cpu"
    print("Using CPU")
print("Using CPU")
device = torch.device("cpu")

model = m.R2UNet(num_classes + 1).to(device)
# model = m.R2UNet(19, weights=F"{weights_path}network_epoch1.pth")

print("Model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")

train_set = d.cityscapesDataset(root=dataset_root_path, split="train")
trainloader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

episode_losses = []
episode_accuracies = []

for e in range(start_epoch, end_epoch):

    episode_loss, episode_accuracy, i = 0, 0, 1

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

        print("\rEpisode", e + 1, "/", end_epoch, "- Batch", i + 1, "/", len(train_set) // batch_size, "\tLoss:", loss,
              "\tAcc:", accuracy, end="")

    '''
    update the general statistics
    Dividing by i is not 100% correct, because the last batch might be smaller
    but for evaluation issues, this can be ignored, because the general trend is
    shown
    '''
    episode_losses.append(episode_loss / i)
    episode_accuracies.append(episode_accuracy / i)
    print("\rEpisode", e + 1, "/", end_epoch, "- Completed \tLoss:", episode_losses[-1], "\tAcc:",
          episode_accuracies[-1])

    # save the models weights
    model.save(F"{weights_path}network_epoch{e}.pth")

plt.title("Loss")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.plot(episode_losses, "r")
plt.legend(loc="right")
plt.savefig("./tas2_losses.png")
plt.show()

plt.title("Accuracy")
plt.xlabel("Episodes")
plt.ylabel("Accuracy")
plt.plot(episode_accuracies, "b")
plt.legend(loc="right")
plt.savefig("./task2_losses.png")
plt.show()
