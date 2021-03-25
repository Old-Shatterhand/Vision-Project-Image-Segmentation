from torch.utils import data

from task1.utils import pascalVOCDataset


def count_values(img, bins=21):
    counts = [0 for _ in range(bins)]
    for j in range(img.shape[0]):
        for i, t in enumerate(img[j, :].bincount(minlength=bins)):
            counts[i] += t.item()
    return counts


local_path = './data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
# dataset variable
# this step takes once around 15 minutes to create the masks
dst = pascalVOCDataset(root=local_path, is_transform=True)

# dataloader variable
trainloader = data.DataLoader(dst, batch_size=1, shuffle=True)

class_counts = [285742327, 3294319, 1407244, 3419625, 2461612, 2427801, 6644208, 5341379, 9872032, 4184171, 3184675,
                4879887, 6162166, 3438894, 4167185, 16969029, 2576183, 3236067, 5255331, 5757808, 3356873]
# class_counts = [0 for _ in range(21)]
if class_counts[0] == 0:
    for j, (image, solution) in enumerate(trainloader):
        if (j + 1) % 10 == 0:
            print("\rImage", j + 1, "/ 1464", end="")
        for i, c in enumerate(count_values(solution[0])):
            class_counts[i] += c

print(class_counts)
print(sum(class_counts))
print([sum(class_counts) / c for c in class_counts])

"""
01 285742327,    
02   3294319,  7 
03   1407244,  1 -
04   3419625,    
05   2461612,  3 
06   2427801,  2 
07   6644208,    
08   5341379,    
09   9872032,    
10   4184171,    
11   3184675,  5
12   4879887,    
13   6162166,    
14   3438894,    
15   4167185,    
16  16969029,    -
17   2576183,  4 
18   3236067,  6 
19   5255331,    
20   5757808,    
21   3356873    
"""
