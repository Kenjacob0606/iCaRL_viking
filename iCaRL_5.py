from time import time

import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from iMNIST import iMNIST
from myNetwork import network
from iCIFAR100 import iCIFAR100
from iCIFAR10 import iCIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt #new

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("using device:", device)
# start_time = time.time()


def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class iCaRLmodel:

    def __init__(self,numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate,dataset,file,train_no):

        super(iCaRLmodel, self).__init__()
        # self.img_size=
        self.file = file
        self.dataset = dataset
        self.train_no = train_no
        self.task_num = 0   #newly added
        self.task_list = []  #newly added
        self.accuracy_list = []  #newly added
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.model = network(numclass,feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass

        
        self.old_model = None #added to store the old model for distillation loss

        self.load_dataset(dataset)

        self.batchsize = batch_size
        self.memory_size=memory_size
        self.task_size=task_size

        self.train_loader=None
        self.test_loader=None

    def load_dataset(self, dataset):
        if dataset == 'CIFAR100':
            # data augmentation for training, testing and class mean computation
            self.transform = transforms.Compose([#transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]) #CIFAR100
            
            self.train_transform = transforms.Compose([#transforms.Resize(img_size),
                                                    transforms.RandomCrop((32,32),padding=4),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ColorJitter(brightness=0.24705882352941178),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]) #CIFAR100
            
            self.test_transform = transforms.Compose([#transforms.Resize(img_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]) #CIFAR100
            
            self.classify_transform=transforms.Compose([
                                                        transforms.RandomHorizontalFlip(p=1.),
                                                        #transforms.Resize(img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]) #CIFAR100
            
            self.train_dataset = iCIFAR100('dataset', transform=self.train_transform, download=True)
            self.test_dataset = iCIFAR100('dataset', test_transform=self.test_transform, train=False, download=True)
            print("CIFAR100 dataset loaded successfully.")

        elif dataset == 'CIFAR10':
            self.transform = transforms.Compose([#transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615))]) #CIFAR10

            self.train_transform = transforms.Compose([#transforms.Resize(img_size),
                                                    transforms.RandomCrop((32,32),padding=4),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ColorJitter(brightness=0.24705882352941178),
                                                    transforms.ToTensor(),      
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615))]) #CIFAR10
            
            self.test_transform = transforms.Compose([#transforms.Resize(img_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615))]) #CIFAR10
            
            self.classify_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                        #transforms.Resize(img_size),
                                                        transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615))]) #CIFAR10

            self.train_dataset = iCIFAR10('dataset', transform=self.train_transform, download=True)                       #CIFAR10
            self.test_dataset = iCIFAR10('dataset', test_transform=self.test_transform, train=False, download=True)
            print("CIFAR10 dataset loaded successfully.")

        elif dataset == 'MNIST':
            self.transform = transforms.Compose([
                                                    transforms.Resize((32,32)),
                                                    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1306, 0.1306, 0.1306), (0.3081, 0.3081, 0.3081))
                                            ]) 

            self.train_transform = transforms.Compose([
                                                    transforms.Resize((32,32)),
                                                    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
                                                    transforms.RandomCrop((32,32),padding=4),
                                                    # transforms.RandomCrop((28),padding=4),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ColorJitter(brightness=0.24705882352941178),
                                                    transforms.ToTensor(),      
                                                    # transforms.Normalize((0.1306), (0.3081))
                                                    transforms.Normalize((0.1306, 0.1306, 0.1306), (0.3081, 0.3081, 0.3081))
                                                    ]) 
            
            self.test_transform = transforms.Compose([
                                                    transforms.Resize((32,32)),
                                                    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1306, 0.1306, 0.1306), (0.3081, 0.3081, 0.3081))
                                                    ]) 
            
            self.classify_transform=transforms.Compose([
                                                        # transforms.RandomHorizontalFlip(p=1.),
                                                    transforms.Resize((32,32)),
                                                    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1306, 0.1306, 0.1306), (0.3081, 0.3081, 0.3081))
                                                    ]) 


            self.train_dataset = iMNIST('MNIST', transform=self.train_transform, download=True)                       
            self.test_dataset = iMNIST('MNIST', test_transform=self.test_transform, train=False, download=True)
            print("MNIST dataset loaded successfully.")
        
        # self.batchsize = batch_size
        # self.memory_size=memory_size
        # self.task_size=task_size

        # self.train_loader=None
        # self.test_loader=None

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        classes=[self.numclass-self.task_size,self.numclass] #select classes for current task
        self.train_loader,self.test_loader=self._get_train_and_test_dataloader(classes) #load data for current task
        if self.numclass>self.task_size:
            self.model.Incremental_learning(self.numclass)  #modify the output layer of the model to accommodate new classes
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader
    
    '''
    def _get_old_model_output(self, dataloader):
        x = {}
        for step, (indexs, imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                old_model_output = torch.sigmoid(self.old_model(imgs))
            for i in range(len(indexs)):
                x[indexs[i].item()] = old_model_output[i].cpu().numpy()
        return x
    '''

    # train model
    # compute loss
    # evaluate model
    def train(self):
        # task_start_time = time.time()  #new
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)       #CIFAR10

        # learning rate is divided by 5 for all tasks during epoch 49 and 63, first task has different lr modification each time than the rest
        for epoch in range(self.epochs):
            if epoch == 48:
                # if self.numclass==self.task_size:
                #      print(1)
                #      opt = optim.SGD(self.model.parameters(), lr=1.0/5, weight_decay=0.00001)
                # else:
                for p in opt.param_groups:
                    p['lr'] =self.learning_rate/ 5
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 5))
            elif epoch == 62:
                # if self.numclass>self.task_size:
                if self.numclass>=self.task_size:
                    for p in opt.param_groups:
                        p['lr'] =self.learning_rate/ 25
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                # else:
                #      opt = optim.SGD(self.model.parameters(), lr=1.0/25, weight_decay=0.00001)
                print("change learning rate:%.3f" % (self.learning_rate / 25))
            elif epoch == 80:
                #   if self.numclass==self.task_size:
                    #  opt = optim.SGD(self.model.parameters(), lr=1.0 / 125,weight_decay=0.00001)
                #   else:
                for p in opt.param_groups:
                    p['lr'] =self.learning_rate/ 125
            #          #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 100))

            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                #output = self.model(images)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()  # compute param gradients by backprop
                opt.step()  # update parameters
                print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            accuracy = self._test(self.test_loader, 1)
            print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
        print(f'Task{self.task_num} complete')  #new edit
        self.task_num += 1      #newly added
        self.task_list.append(self.task_num)  #newly added
        return accuracy

    def _test(self, testloader, mode):
        # task = 0  # new
        if mode==0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for step, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        # task += 1  # new
        self.model.train()
        return accuracy


    def _compute_loss(self, indexs, imgs, target):
        output=self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            #old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            ##Distillation loss
            old_target=torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)


    # change the size of examplar
    def afterTrain(self,accuracy):
        self.model.eval()
        torch.cuda.empty_cache()
        m=int(self.memory_size/self.numclass) #compute exemplar size for each class
        self._reduce_exemplar_sets(m) #exemplars per class decrease as more classes are added
        for i in range(self.numclass-self.task_size,self.numclass): #loop over newly learned classes only
            print('construct class %s examplar:'%(i),end='')
            images=self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images,m)
        self.numclass+=self.task_size
        self.compute_exemplar_class_mean()
        self.model.train()
        KNN_accuracy=self._test(self.test_loader,0)
        print("NMS accuracy："+str(KNN_accuracy.item()))
        filename = f'cifar100_lr=2.1_def/model/{self.task_num}-accuracy-{accuracy:.3f}_KNN_accuracy-{KNN_accuracy:.3f}_increment-{i + 10}_net_train={self.train_no}.pkl'  # changed : to -
        self.accuracy_list.append(KNN_accuracy)  #newly added
        filename2 = f'cifar100_lr=2.1_def/model/model_class_mean_{self.task_num}_train={self.train_no}.pth'  #newly added
        torch.save(self.model,filename)
        torch.save({'class_mean_set': self.class_mean_set,}, filename2)
        if self.old_model is not None:      #CIFAR10
            self.old_model.cpu()
            del self.old_model
            torch.cuda.empty_cache()
        self.old_model = torch.load(filename, weights_only=False)        # added weights_only parameter
        self.old_model.to(device)
        self.old_model.eval()
        
#GRAPHS
        if self.task_num==10:                    #Changable param depending on how many class each task
            if self.task_num==10 and self.dataset=='MNIST':     
                plt.plot(self.task_list, self.accuracy_list, "g+")
                plt.xticks(range(len(self.accuracy_list)+1))
                # plt.yticks(range(0, 101, 10))
                plt.xlabel("Task")
                plt.ylabel("Accuracy")
                plt.title("Accuracy vs Tasks")
                plt.savefig('CIFAR10_class=1_mem=500_def/model/accuracy_vs_tasks.png') 
                # plt.show()
                plt.plot(self.task_list, self.accuracy_list, "g+-")
                plt.xticks(range(len(self.accuracy_list)+1))
                # plt.yticks(range(0, 101, 10))
                plt.xlabel("Task")
                plt.ylabel("Accuracy")
                plt.title("Accuracy vs Tasks")
                plt.savefig('CIFAR10_class=1_mem=500_def/model/accuracy_vs_tasks_line.png') 
            else:
                plt.plot(self.task_list, self.accuracy_list, "g+")
                plt.xticks(range(len(self.accuracy_list)+1))
                plt.yticks(range(0, 101, 10))
                plt.xlabel("Task")
                plt.ylabel("Accuracy")
                plt.title("Accuracy vs Tasks")
                plt.savefig(f'cifar100_lr=2.1_def/model/accuracy_vs_tasks_train={self.train_no}.png') 
                # plt.show()
                plt.plot(self.task_list, self.accuracy_list, "g+-")
                plt.xticks(range(len(self.accuracy_list)+1))
                plt.yticks(range(0, 101, 10))
                plt.xlabel("Task")
                plt.ylabel("Accuracy")
                plt.title("Accuracy vs Tasks")
                plt.savefig(f'cifar100_lr=2.1_def/model/accuracy_vs_tasks_line_train={self.train_no}.png') 
                # plt.show()
                # print(len(self.class_mean_set))
                plt.clf()
    
#GRAPHS


    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))     
        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index]) #newly edited

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)
        #self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))



    def Image_transform(self, images, transform): 
        if isinstance(images,torch.Tensor):
            images = images.cpu().numpy()
        if(isinstance(images,list)):    #images when called by compute_exemlar_class_mean is a list of tensors for MNIST
            for i in range(len(images)):
                if isinstance(images[i],torch.Tensor):
                    image = images[i].cpu().numpy()
                    images[i] = image
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)    
            #latest edit (self.transform -> transform), might be accidental bug in orginial impl, cause in the case of exemplar set mean computation
            # it uses transform passed in for only first image (which would be self.classify_transform) and self.transform for the rest of the images,
            # which is inconsistent and might cause performance drop. in the other cases it passes in self.transform as transform, hence no inconsistency. 
        return data

    def compute_class_mean(self, images, transform):   
        x = self.Image_transform(images, transform).to(device)
        with torch.no_grad():
            feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy() #extract features, detaach from graph, normalize to unit length, move to cpu and convert to numpy
            # del x
            # torch.cuda.empty_cache()
        #feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output


    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s"%(str(index)))
            exemplar=self.exemplar_set[index]
            #exemplar=self.train_dataset.get_image_class(index)
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_,_=self.compute_class_mean(exemplar,self.classify_transform)
            class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def classify(self, test):
        result = []
        test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        #test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)
