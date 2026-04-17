from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
from ResNet import resnet18_MNIST_cbam
from ResNet import resnet34_cbam
from ResNet import resnet50_cbam
import torch
import time

#"cifar10_class=1_mem=5K_def"

for train_no in range(3,4):
    numclass=1      #num of classes learned initially, will be updated in incremental learning   
    feature_extractor=resnet18_cbam(num_classes=numclass) #try other resnets
    img_size=32
    batch_size=128  
    task_size=1      #num of classes learned each task
    memory_size= 5000
    epochs=70 #was 100
    learning_rate=2.0
    file=1
    dataset='CIFAR10' #try other dataset
    train_no = 1
    filenames = "cifar10_class=1_mem=5K_def"

    model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate,dataset,file,train_no,filenames) #try other dataset
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
        filename = f'{filenames}/model/task_{i}_training_time={task_end_time - task_start_time:.2f}_train={train_no}.txt'     #modify
        torch.save((task_end_time - task_start_time), filename)
        # if i==9:
    end_time = time.time()

    # print('Total training time: {:.2f} seconds'.format(end_time - start_time))
    filename2 = f'{filenames}/model/total_training_time= {end_time - start_time:.2f}_train={train_no}.txt'        #modify
    torch.save((end_time - start_time), filename2)

    del model
    torch.cuda.empty_cache()

######################################################################################################################################################
#TRAIN 2

# numclass=1      #num of classes learned initially, will be updated in incremental learning
# feature_extractor=resnet18_cbam(num_classes=numclass) #try other resnets
# img_size=32
# batch_size=128  
# task_size=1 #num of classes learned each task
# memory_size= 5000
# epochs=70 #was 100
# learning_rate=2.0
# file=1
# dataset='CIFAR10' #try other dataset
# train_no = 2
# filenames = "cifar10_class=1_mem=5K_def"

# model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate,dataset,file,train_no,filenames) #try other dataset
# #model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

# start_time = time.time()
# for i in range(10): #was 10,5
#     # if i==0:
#     #     start_time = time.time()
#     task_start_time = time.time()
#     model.beforeTrain()
#     accuracy=model.train()
#     model.afterTrain(accuracy)
#     task_end_time = time.time()
#     filename = f'{filenames}/model/task_{i}_training_time={task_end_time - task_start_time:.2f}_train={train_no}.txt'     #modify
#     torch.save((task_end_time - task_start_time), filename)
#     # if i==9:
# end_time = time.time()

# # print('Total training time: {:.2f} seconds'.format(end_time - start_time))
# filename2 = f'{filenames}/model/total_training_time= {end_time - start_time:.2f}_train={train_no}.txt'        #modify
# torch.save((end_time - start_time), filename2)

# del model
# torch.cuda.empty_cache()
# ######################################################################################################################################################
# #TRAIN 3

# numclass=1          #num of classes learned initially, will be updated in incremental learning
# feature_extractor=resnet18_cbam(num_classes=numclass) #try other resnets
# img_size=32
# batch_size=128  
# task_size=1          #num of classes learned each task
# memory_size= 5000
# epochs=70 #was 100
# learning_rate=2.0
# file=1
# dataset='CIFAR10' #try other dataset
# train_no = 3
# filenames = "cifar10_class=1_mem=5K_def"

# model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate,dataset,file,train_no,filenames) #try other dataset

# start_time = time.time()
# for i in range(10): #was 10,5
#     # if i==0:
#     #     start_time = time.time()
#     task_start_time = time.time()
#     model.beforeTrain()
#     accuracy=model.train()
#     model.afterTrain(accuracy)
#     task_end_time = time.time()
#     filename = f'{filenames}/model/task_{i}_training_time={task_end_time - task_start_time:.2f}_train={train_no}.txt'     #modify
#     torch.save((task_end_time - task_start_time), filename)
#     # if i==9:
# end_time = time.time()

# # print('Total training time: {:.2f} seconds'.format(end_time - start_time))
# filename2 = f'{filenames}/model/total_training_time= {end_time - start_time:.2f}_train={train_no}.txt'        #modify
# torch.save((end_time - start_time), filename2)

