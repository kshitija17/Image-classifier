# Imports here

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import seaborn as sb

import matplotlib.pyplot as plt


import argparse
def main():
    parser=argparse.ArgumentParser(description='train network')
    parser.add_argument('data_directory', type=str, help='dataset directory',default='flowers')
    parser.add_argument('--save_dir',type=str, help='enter dictionary name to save checkpoint')
    parser.add_argument('--arch', type=str, help='enter architecture name(vgg13 or vgg16)',default='vgg16')
    parser.add_argument('--learning_rate', type=int, help='enter learning rate',default=0.001)
    parser.add_argument('--hidden_units',type=int, help='enter hidden units',default=600)
    parser.add_argument('--epochs', type=int, help='enter no. of epochs',default=8)
    parser.add_argument('--gpu', type=bool, help='use gpu',default=True)

    args=parser.parse_args()

    #directory paths
    data_dir=args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                         ])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                         ])

    vald_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                         ])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    vald_dataset = datasets.ImageFolder(valid_dir,transform=vald_transforms)
    test_dataset = datasets.ImageFolder(test_dir,transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valdloader=torch.utils.data.DataLoader(vald_dataset,batch_size=64)
    testloader=torch.utils.data.DataLoader(test_dataset,batch_size=64)


    model=train(args.arch,args.hidden_units,args.learning_rate,args.epochs,trainloader,args.gpu,valdloader)
    find_accuracy(model,testloader,args.gpu)
    save_checkpoint(args.save_dir,model,args.hidden_units,args.arch,train_dataset)


def validation(model,valdloader,criterion):
    valid_loss=0
    accuracy=0
    for images,labels in valdloader:
        images=images.cuda(async=True)
        labels=labels.cuda(async=True)
        output=model.forward(images)
        valid_loss+=criterion(output,labels).item()
        p=torch.exp(output)
        equality=(labels.data==p.max(dim=1)[1])
        accuracy+=equality.type(torch.FloatTensor).mean()
        #print(p)
    return valid_loss,accuracy



   # TODO: Build and train your network
#load model
def train(arch,hidden_units,learnrate,epochs,trainloader,gpu,valdloader):
    if arch=='vgg13':
        model=models.vgg13(pretrained=True)
    else:
        model=models.vgg16(pretrained=True)

    #freeze parameters
    for p in model.parameters():
        p.requires_grad=False

    #create classifier
    input_size=25088
    hidden_size=hidden_units
    output_size=102

    classifier=nn.Sequential(nn.Linear(input_size,hidden_size),
                            nn.ReLU(),
                            nn.Dropout(p=0.25),
                            nn.Linear(hidden_size,output_size),
                            nn.LogSoftmax(dim=1))
    model.classifier=classifier

    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr=learnrate)

#train a network
    print_every=20
    steps=0

    if gpu==False:
        model.to('cpu')
    else:
        model.to('cuda')

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (images, labels) in enumerate(trainloader):
            #print(images.size())
            steps += 1
            if gpu==True:
                images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss,accuracy=validation(model,valdloader,criterion)

                print("Epoch NO: {}/{}".format(e+1,epochs),
                  "Train Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss:{:.4f}".format(validation_loss/len(valdloader)),
                   "Valid Accuracy:{:.4f}".format(accuracy/len(valdloader)))



                running_loss = 0
                model.train()
    return model

# TODO: Do validation on the test set
def find_accuracy(model,dataloader_name,gpu):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader_name:
            images, labels = data
            if gpu==True:
                images=images.cuda(async=True)
                labels=labels.cuda(async=True)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test dataset: %d %%' % (100 * correct / total))

# TODO: Save the checkpoint
def save_checkpoint(save_dir,model,hidden_units,arch,train_dataset):
    model.class_to_idx=train_dataset.class_to_idx
    checkpoint={'input_size':25088,
                'output_size':102,
                'hidden_layers':hidden_units,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx,
                'architecture':arch
                }
    if save_dir is not None:
        torch.save(checkpoint,save_dir+'/checkpoint.pth')
    else:
        torch.save(checkpoint,'checkpoint.pth')

if __name__=='__main__':
    main()
