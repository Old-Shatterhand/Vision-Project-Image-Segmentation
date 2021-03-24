import os
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import colors
from torch.utils import data
from torchvision import transforms


class cityscapesDataset(data.Dataset):
    """
    We copied this container from the given code of task 1 and modified it according to the cityscapes dataset
    """

    def __init__(self, root, split="train", img_size=(512, 256), test_mode=False):
        self.root = root
        self.split = split
        self.test_mode = test_mode
        self.num_classes = 19
        self.files = {"train": [], "test": [], "val": []}
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.encode = False

        self.tf = transforms.Compose(
            [
                # add more transformations as you see fit
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]
        )

        # iterate over the splits
        for split in ["train", "val", "test"]:
            path = pjoin(self.root, "gtFine", split)

            # iterate over the cities included in the directory
            for directory in os.listdir(path):
                if not self.encode and not os.path.exists(pjoin(path, directory, "pre_encoded_" + str(img_size[0]))):
                    self.encode = True
                    os.mkdir(pjoin(path, directory, "pre_encoded_" + str(img_size[0])))
                    os.mkdir(pjoin(self.root, "leftImg8bit", split, directory, "pre_encoded_" + str(img_size[0])))

                # iterate over all imaged from the city
                for i, image in enumerate(os.listdir(pjoin(path, directory))):
                    if "color" in image:
                        lbl_path = pjoin(path, directory, image)
                        self.files[split].append(pjoin(path, directory, "pre_encoded_" + str(img_size[0]), image.replace("png", "pt")))
                        if self.encode:
                            print("\rPreprocessing:", split, "- Image", i, end="")
                            img_path = lbl_path.replace("gtFine", "leftImg8bit").replace("_color", "")
                            img = Image.open(img_path).convert("RGB")
                            lbl = Image.open(lbl_path).convert("RGB")

                            img, lbl = self.transform(img, lbl)
                            lbl = torch.clamp(lbl, max=self.num_classes)

                            torch.save(img, pjoin(self.root, "leftImg8bit", split, directory, "pre_encoded_" + str(img_size[0]), image.replace("png", "pt")).replace("gtFine", "leftImg8bit").replace("_color", ""))
                            torch.save(lbl, self.files[split][-1])

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        lbl_path = self.files[self.split][index]
        img_path = lbl_path.replace("gtFine", "leftImg8bit").replace("_color", "")
        # im = Image.open(img_path).convert("RGB")
        # lbl = Image.open(lbl_path).convert("RGB")

        # im, lbl = self.transform(im, lbl)
        # return im, torch.clamp(lbl, max=self.num_classes)
        lbl = torch.load(lbl_path)
        img = torch.load(img_path)
        return img, lbl

    def transform(self, img, lbl):
        # uint8 with RGB mode
        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]))  # , interpolation_mode=PIL.Image.NEAREST)

        img = self.tf(img)
        lbl = torch.from_numpy(self.encode_segmap(np.array(lbl))).long()

        lbl[lbl == 255] = 0
        return img, lbl

    def encode_segmap(self, mask):
        """
        Encode segmentation label images as cityscapes classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        # print("Label-Sum:", np.sum(mask))

        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_cs_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_cs_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.num_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    @staticmethod
    def get_cs_labels():
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (19, 3)
        """
        return np.asarray(
            [[128, 64, 128],
             [244, 35, 232],
             [70, 70, 70],
             [102, 102, 156],
             [190, 153, 153],
             [153, 153, 153],
             [250, 170, 30],
             [220, 220, 0],
             [107, 142, 35],
             [152, 251, 152],
             [0, 130, 180],
             [220, 20, 60],
             [255, 0, 0],
             [0, 0, 142],
             [0, 0, 70],
             [0, 60, 100],
             [0, 80, 100],
             [0, 0, 230],
             [119, 11, 32],
             [0, 0, 0]]
        )


'''
d = cityscapesDataset("C:/Users/joere/Desktop/data", split="val")
img, lbl = d[0]

cmap = colors.ListedColormap(d.get_cs_labels() / 255)
bounds = list(range(19))
norm = colors.BoundaryNorm(bounds, cmap.N)

fig = plt.figure()
im1 = fig.add_subplot(211)
im1.imshow(img.transpose(0, 2).transpose(0, 1).numpy())

im2 = fig.add_subplot(212)
im2.imshow(lbl.numpy().astype('uint8'), cmap=cmap, norm=norm)
plt.show()
'''

lbl = torch.load("C:/Users/joere/Desktop/zurich_000054_000019_gtFine_color.pt")
cmap = colors.ListedColormap(cityscapesDataset.get_cs_labels() / 255)
bounds = list(range(19))
norm = colors.BoundaryNorm(bounds, cmap.N)

plt.imshow(lbl.numpy().astype('uint8'), cmap=cmap, norm=norm)
plt.show()

print("Finished")

