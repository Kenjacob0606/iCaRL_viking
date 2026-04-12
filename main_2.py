from iCaRL_2 import iCaRLmodel
from ResNet import resnet18_cbam
from ResNet import resnet18_MNIST_cbam
from ResNet import resnet34_cbam
from ResNet import resnet50_cbam
import torch
import time

#CIFAR10_lr=1.5_mem=750_class=1_def

numclass=1#num of classes learned initially, will be updated in incremental learning
feature_extractor=resnet18_cbam() #try other resnets
img_size=32
batch_size=128  
task_size=1 #num of classes learned each task
memory_size= 750
epochs=70 #was 100
learning_rate=1.5
file=1
dataset='CIFAR10' #try other dataset
train = 1

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate,dataset,file,train) #try other dataset
#model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

start_time = time.time()
for i in range(10): #was 10,5
    # if i==0:
    #     start_time = time.time()
    task_start_time = time.time()
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)
    task_end_time = time.time()
    filename = f'CIFAR10_lr=1.5_mem=750_class=1_def/model/task_{i}_training_time={task_end_time - task_start_time:.2f}_train={train}.txt'
    torch.save((task_end_time - task_start_time), filename)
    # if i==9:
end_time = time.time()

# print('Total training time: {:.2f} seconds'.format(end_time - start_time))
filename2 = f'CIFAR10_lr=1.5_mem=750_class=1_def/model/total_training_time= {end_time - start_time:.2f}_train={train}.txt'
torch.save((end_time - start_time), filename2)

#####################################################################################################################################

#TRAIN2

numclass=1#num of classes learned initially, will be updated in incremental learning
feature_extractor=resnet18_cbam() #try other resnets
img_size=32
batch_size=128  
task_size=1 #num of classes learned each task
memory_size= 750
epochs=70 #was 100
learning_rate=1.5
file=1
dataset='CIFAR10' #try other dataset
train = 2

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate,dataset,file,train) #try other dataset
#model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

start_time = time.time()
for i in range(10): #was 10,5
    # if i==0:
    #     start_time = time.time()
    task_start_time = time.time()
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)
    task_end_time = time.time()
    filename = f'CIFAR10_lr=1.5_mem=750_class=1_def/model/task_{i}_training_time={task_end_time - task_start_time:.2f}_train={train}.txt'
    torch.save((task_end_time - task_start_time), filename)
    # if i==9:
end_time = time.time()

# print('Total training time: {:.2f} seconds'.format(end_time - start_time))
filename2 = f'CIFAR10_lr=1.5_mem=750_class=1_def/model/total_training_time= {end_time - start_time:.2f}_train={train}.txt'
torch.save((end_time - start_time), filename2)



