from iCaRL_MNIST_15epo import iCaRLmodel
from ResNet import resnet18_cbam
from ResNet import resnet18_MNIST_cbam
from ResNet import resnet34_cbam
from ResNet import resnet50_cbam
import torch
import time

#   "MNIST_class=1_mem=1000_def"

for train_no in range (1,4):
    dataset='MNIST' #try other dataset
    numclass=1      #num of classes learned initially, will be updated in incremental learning
    if dataset == 'CIFAR100':
        numclasses = 100
    else:
        numclasses = 10
    feature_extractor=resnet18_cbam(num_classes=numclasses) #try other resnets
    img_size=32
    batch_size=128  
    task_size=1         #num of classes learned each task
    memory_size= 500
    epochs=15 #was 100
    learning_rate= 1.5
    file=1
    filenames = "MNIST_mem=500_def"


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

