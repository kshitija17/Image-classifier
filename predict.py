import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import argparse

def main():
    parser=argparse.ArgumentParser(description='predict image')
    parser.add_argument('image_path', type=str, help='Enter image path')
    parser.add_argument('--top_k', type=int, help='top k probabilities', default=5)
    parser.add_argument('checkpoint', type=str, help='checkpoint file name',default='checkpoint.pth')
    parser.add_argument('--category_names', type=str, help='cat_to_name json code file',default='cat_to_name.json')
    parser.add_argument('--gpu', type=bool, help='use gpu',default=True)
    args=parser.parse_args()


    #json code here
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model=load_checkpoint(args.checkpoint,args.gpu)
    display_fun(args.image_path,model,args.top_k,args.gpu,cat_to_name)

def load_checkpoint(filepath,gpu):
    checkpoint=torch.load(filepath)
    if checkpoint['architecture']=='vgg13':
        model=models.vgg13(pretrained=True)
    else:
        model=models.vgg16(pretrained=True)

    for p in model.parameters():
        p.requires_grad=False
    if gpu==True:
        model.to('cuda')
    classifier=nn.Sequential(nn.Linear(checkpoint['input_size'],checkpoint['hidden_layers']),
                        nn.ReLU(),
                        nn.Dropout(p=0.25),
                        nn.Linear(checkpoint['hidden_layers'],checkpoint['output_size']),
                        nn.LogSoftmax(dim=1))
    model.classifier=classifier

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint['class_to_idx']

    return model



#Image preprocessing
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    image=Image.open(image_path)

    #resize image
    if image.size[0]>image.size[1]:
        image.thumbnail((100000,256))
    else:
        image.thumbnail((256,100000))
    #margins
    left=(image.width-224)/2
    right=224+left
    bottom=(image.height-224)/2
    top=224+bottom

    #CenterCrop
    image=image.crop((left,bottom,right,top))

    #color channel
    np_image=np.array(image)
    np_image=np_image/255

    #normalization
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    #print(mean)
    #print(std)

    np_image=(np_image-mean)/std

    #changing dimension
    np_image=np_image.transpose((2,0,1))

    return np_image


#Class prediction
def predict(image_path, model, top_k_no,gpu,cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    if gpu==True:
        model.to('cuda')
    image=process_image(image_path)
    image=torch.from_numpy(image).type(torch.FloatTensor)

    image.unsqueeze_(0)

    image=image.to('cuda')

    pb=torch.exp(model.forward(image))

    #print(pb)
    prob,label=pb.topk(top_k_no)
    prob=prob.cpu()
    label=label.cpu()
    prob=prob.detach().numpy().tolist()[0]
    label=label.detach().numpy().tolist()[0]

    idx_to_class={v:k for k, v in model.class_to_idx.items()}
    flowers = [cat_to_name[idx_to_class[lab]] for lab in label]

    return flowers,prob

#Sanity Checking
# TODO: Display an image along with the top k classes
def display_fun(image_path,model,topk,gpu,cat_to_name):
    if gpu==True:
        model.to('cuda')

    flower_no=image_path.split('/')[2]
    title_name=cat_to_name[flower_no]
    image=process_image(image_path)
    flowers,probs=predict(image_path,model,topk,gpu,cat_to_name)

    for i in range(topk):
        print("Flower name: {:<20} Probability: {:.4f}".format(flowers[i],probs[i]))



if __name__=='__main__':
    main()
