# from torch import tensor
import torch
import torch
from torchvision.datasets import CIFAR10
import numpy as np
from PIL import Image


class iCIFAR10(CIFAR10):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR10, self).__init__(root,
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

        self.ExemplarData = []
        self.ExemplarLabels = []
        self.NewClassData = []
        self.NewClassLabels = []
        self.exemplar_size = 0

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes, offset=0):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            actual_label = label - offset #After training on cifar 10, the labels for MNIST will start from 10, so we need to subtract 10 to get the actual label
            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas if self.TestData is None else np.concatenate(
            (self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels is None else np.concatenate(
            (self.TestLabels, labels), axis=0)
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def getTrainData(self, classes, exemplar_set, offset=0):

        # datas, labels = [], []
        self.ExemplarData, self.ExemplarLabels, self.NewClassData, self.NewClassLabels = [], [], [], []

        if len(exemplar_set) != 0:
            for label, exemplar in enumerate(exemplar_set):
                for img in exemplar:
                    self.ExemplarData.append(img)
                    self.ExemplarLabels.append(label)
            self.exemplar_size = len(self.ExemplarData)
            # for exemplar in exemplar_set:
                # exemplar_np = [img.numpy() if isinstance(img, torch.Tensor) else img for img in exemplar]
                # datas.append(np.array(exemplar_np))
                # datas.append(exemplar_np)
            # length = len(datas[0])
            # self.exemplar_size = length * len(exemplar_set) #new
            # labels = [np.full((length), label)
            #           for label in range(len(exemplar_set))]

        for label in range(classes[0], classes[1]):
            actual_label = label - offset           #After training on cifar 10, the labels for MNIST will start from 10, so we need to subtract 10 to get the actual label
            data = self.data[np.array(self.targets) == actual_label]
            for img in data:
                self.NewClassData.append(img)
                self.NewClassLabels.append(label)
            # datas.append(data)
            # labels.append(np.full((data.shape[0]), label))
        # self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        self.TrainData, self.TrainLabels = self.ExemplarData + self.NewClassData, self.ExemplarLabels + self.NewClassLabels
        # print("the size of train set is %s" % (str(self.TrainData.shape)))
        # print("the size of train label is %s" % str(self.TrainLabels.shape))
        print("the size of train set is %s" % (str(len(self.TrainData))))
        print("the size of train label is %s" % (str(len(self.TrainLabels))))


    def getTrainItem(self, index):
        # if isinstance(self.TrainData, torch.Tensor):
        #     self.TrainData = self.TrainData.cpu().numpy()
        # if isinstance(self.TrainLabels, torch.Tensor):
            # self.TrainLabels = self.TrainLabels.cpu().numpy()
        # img, target = Image.fromarray(
        #     self.TrainData[index]), self.TrainLabels[index]
        target = self.TrainLabels[index]

        if index < self.exemplar_size: #new 
            img = self.ExemplarData[index]
            # print(f"exemplar index {index}, shape: {img.shape}") 

        else: 
            raw = self.NewClassData[index - self.exemplar_size]
            img = Image.fromarray(raw) 
            if self.transform:
                img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return index, img, target

    def getTestItem(self, index):
        if isinstance(self.TestData, torch.Tensor):
            self.TestData = self.TestData.cpu().numpy()
        if isinstance(self.TestLabels, torch.Tensor):
            self.TestLabels = self.TestLabels.cpu().numpy()
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

    def get_image_class(self, label, offset=0):     #addde offset
        actual_label = label - offset
        return self.data[np.array(self.targets) == actual_label]
