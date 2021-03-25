import os
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class cityscapesDataset(data.Dataset):
    """
    We copied this container from the given code of task 1 and modified it according to the cityscapes dataset
    """

    def __init__(self, root, split="train", img_size=(512, 256)):
        """
        Initialization of the cityscapes-dataset contained
        :param root: root directory of the dataset, it should then contain two folders called leftImg8bit and gtFine
        containing the images (https://www.cityscapes-dataset.com/file-handling/?packageID=3) and the according fine
        annotations (https://www.cityscapes-dataset.com/file-handling/?packageID=1). Attention, both are download links.
        :param split: name of the split to load in this dataset. Default is "train".
        :param img_size: size of the image to use for outputting. Original is (2048, 1024), default is (512, 256).
        """
        self.root = root
        self.split = split
        self.num_classes = 19
        self.files = {"train": [], "test": [], "val": []}
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        self.tf = transforms.Compose(
            [
                # add more transformations as you see fit
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]
        )

        # iterate over the splits and check if all files for all splits and the given image size are precomputed
        for split in ["train", "val", "test"]:
            path = pjoin(self.root, "gtFine", split)

            # iterate over the cities included in the directory
            for city in os.listdir(path):
                encode = False
                if not os.path.exists(pjoin(path, city, "pre_encoded_" + str(img_size[0]))):
                    # if the folders containing the preprocessed images does not exist, create them and set a flag to
                    # compute all according images
                    encode = True
                    os.mkdir(pjoin(path, city, "pre_encoded_" + str(img_size[0])))
                    os.mkdir(pjoin(self.root, "leftImg8bit", split, city, "pre_encoded_" + str(img_size[0])))
                    print(split, "-", city, "- Folders exist:",
                          os.path.exists(pjoin(path, city, "pre_encoded_" + str(img_size[0]))) and os.path.exists(
                              pjoin(self.root, "leftImg8bit", split, city, "pre_encoded_" + str(img_size[0]))))

                # iterate over all imaged from the city
                for i, image in enumerate(os.listdir(pjoin(path, city))):
                    if "color" in image:
                        lbl_path = pjoin(path, city, image)
                        self.files[split].append(
                            pjoin(path, city, "pre_encoded_" + str(img_size[0]), image.replace("png", "pt")))

                        if encode:
                            # if necessary precompute the image transformation and label transformations. Then save them
                            # to be loaded later on to save computational time
                            img_path = lbl_path.replace("gtFine", "leftImg8bit").replace("_color", "")
                            img = Image.open(img_path).convert("RGB")
                            lbl = Image.open(lbl_path).convert("RGB")

                            img, lbl = self.transform(img, lbl)
                            lbl = torch.clamp(lbl, max=self.num_classes)

                            img_save_path = pjoin(self.root, "leftImg8bit", split, city,
                                                  "pre_encoded_" + str(img_size[0]),
                                                  image.replace("png", "pt")).replace("gtFine", "leftImg8bit").replace(
                                "_color", "")
                            torch.save(img, img_save_path)
                            torch.save(lbl, self.files[split][-1])
                            print("\r\tFiles", i, "exist:",
                                  os.path.exists(img_save_path) and os.path.exists(self.files[split][-1]), end="")
                if encode:
                    print()

    def __len__(self):
        """
        The length of the dataset is defined as the number of samples in the split used
        :return: number of sampels in the used split
        """
        return len(self.files[self.split])

    def __getitem__(self, index):
        """
        Return an sample from the split
        :param index: index of the sample
        :return: sample from the split as tuple of image and label (solution)
        """
        lbl_path = self.files[self.split][index]
        return torch.load(lbl_path.replace("gtFine", "leftImg8bit").replace("_color", "")), torch.load(lbl_path)

    def transform(self, img, lbl):
        """
        Perform the transformations and resizing on the images
        :param img: image to transform
        :param lbl: label to transform
        :return: tuple of two torch tensors with the image and the label
        """
        # uint8 with RGB mode
        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]))

        img = self.tf(img)
        lbl = torch.from_numpy(self.encode_segmap(np.array(lbl))).long()

        lbl[lbl == 255] = 0
        return img, lbl

    def encode_segmap(self, mask):
        """
        Encode segmentation label images as cityscapes classes
        :param mask: raw segmentation label image of dimension (M, N, 3), in which the Pascal classes are encoded as
        colors.
        :returns: class map with dimensions (M,N), where the value at a given location is the integer denoting the class
        index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_cs_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """
        Decode segmentation class labels into a color image
        :param label_mask: an (M,N) array of integer values denoting the class label at each spatial location.
        :param plot: whether to show the resulting color image in a figure.
        :return: the resulting decoded color image.
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
        """
        Load the mapping that associates pascal classes with label colors
        Returns: np.ndarray with dimensions (19, 3)
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
