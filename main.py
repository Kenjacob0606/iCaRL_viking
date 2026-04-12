from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
from ResNet import resnet18_MNIST_cbam
from ResNet import resnet34_cbam
from ResNet import resnet50_cbam
import torch
import time

#class=1_mem=200_def_2ndtrain

numclass=5#num of classes learned initially, will be updated in incremental learning
feature_extractor=resnet18_cbam() #try other resnets
img_size=32
batch_size=128  
task_size=5 #num of classes learned each task
memory_size= 2000
epochs=70 #was 100
learning_rate=2.0
file=1
dataset='CIFAR100' #try other dataset

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate,dataset,file) #try other dataset
#model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

start_time = time.time()
for i in range(20): #was 10,5
    # if i==0:
    #     start_time = time.time()
    task_start_time = time.time()
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)
    task_end_time = time.time()
    filename = f'cifar100_class=5_def/model/task_{i}_training_time= {task_end_time - task_start_time:.2f}.txt'
    torch.save((task_end_time - task_start_time), filename)
    # if i==9:
end_time = time.time()

# print('Total training time: {:.2f} seconds'.format(end_time - start_time))
filename2 = f'cifar100_class=5_def/model/total_training_time= {end_time - start_time:.2f}.txt'
torch.save((end_time - start_time), filename2)




