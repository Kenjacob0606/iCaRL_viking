from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST #hand written digits
from torchvision.datasets import FashionMNIST #fashion products
from torchvision.datasets import SVHN #complex digits
from torchvision.datasets import CelebA #celebrity faces
from torchvision.datasets import ImageNet #natural images
import numpy as np
from PIL import Image


class iDataset(CIFAR10):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iDataset, self).__init__(root,
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)

        self.target_test_transform = target_test_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        # self.TestData = []
        self.TestData = None
        self.TestLabels = None

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas if self.TestData is None else np.concatenate(
            (self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels is None else np.concatenate(
            (self.TestLabels, labels), axis=0)
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def getTrainData(self, classes, exemplar_set):

        datas, labels = [], []
        if len(exemplar_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label)
                      for label in range(len(exemplar_set))]

        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))

    def getTrainItem(self, index):
        img, target = Image.fromarray(
            self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return index, img, target

    def getTestItem(self, index):
        img, target = Image.fromarray(
            self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        if self.target_test_transform:
            target = self.target_test_transform(target)

        return index, img, target

    # def __getitem__(self, index):
    #     if self.TrainData != []:
    #         return self.getTrainItem(index)
    #     elif self.TestData != []:
    #         return self.getTestItem(index)

    def __getitem__(self, index):
        if len(self.TrainData) > 0:
            return self.getTrainItem(index)
        elif len(self.TestData) > 0:
            return self.getTestItem(index)
        else:
            raise RuntimeError("Dataset is empty.")

    # def __len__(self):
    #     if self.TrainData!=[]:
    #         return len(self.TrainData)
    #     elif self.TestData!=[]:
    #         return len(self.TestData)

    def __len__(self):
        if self.TrainData is not None and len(self.TrainData) > 0:
            return len(self.TrainData)
        elif self.TestData is not None and len(self.TestData) > 0:
            return len(self.TestData)
        else:
            return 0

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]
