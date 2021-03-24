import dataloader as d
import torch
from torch.utils import data


def count_values(img, bins=20):
    counts = [0 for _ in range(bins)]
    for j in range(img.shape[0]):
        for i, t in enumerate(img[j, :].bincount(minlength=bins)):
            counts[i] += t.item()
    return counts

project_root_dir = "../../datasets/"
dataset_root_path = project_root_dir + "data/cityscapes/"
batch_size = 1
num_classes = 20
total_class_counts = [0 for _ in range(num_classes)]
for i, s in enumerate(["train", "val", "test"]):
    dataset = d.cityscapesDataset(root=dataset_root_path, split=s)
    loader = data.DataLoader(dataset, batch_size=batch_size)
    class_counts = [0 for _ in range(num_classes)]

    for j, (image, solution) in enumerate(loader):
        if (j + 1) % 10 == 0:
            print("\rSplit", s, "- Image", j + 1, "/", len(dataset), end="")

        for k, c in enumerate(count_values(solution[0])):
            class_counts[k] += c
            total_class_counts[k] += c
    print(s + ":")
    print(class_counts)
    print([str(c / sum(class_counts)) for c in class_counts])

print("Total:")
print(total_class_counts)
print([str(c / sum(total_class_counts)) for c in total_class_counts])

