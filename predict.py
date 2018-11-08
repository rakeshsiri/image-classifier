import time
from collections import OrderedDict
from PIL import Image
import numpy as np
import argparse
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

#Reading inputs from cmd prompt
parser = argparse.ArgumentParser(add_help=True)

parser.add_argument('img_path',help="input path of image")
parser.add_argument('checkpoint',help="checkpoint path of saved model")
parser.add_argument('--category_names', required=False, help="category mapping json files")
parser.add_argument('--top_k', required=False, type=int, default=3, help="Return top KK most likely classes")
parser.add_argument('--gpu', required=False, action="store_true",help="Use GPU to compute")

args = parser.parse_args()

img_path = args.img_path
checkpoint = args.checkpoint
category_names = args.category_names
top_k = args.top_k
use_gpu = args.gpu

# Label mapping
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
    
# Function that loads a checkpoint and rebuilds the model
def load_model_from_path(checkpoint_path):
    checkpoint_ld=torch.load(checkpoint_path)
    criteria=checkpoint_ld['criteria']
    optimiz=checkpoint_ld['optimizer_state']
    scheduler = checkpoint_ld['scheduler']

    if checkpoint_ld['arch']=='vgg16':
        model_ld = models.vgg16(pretrained=True)
    else:
        model_ld = models.vgg13(pretrained=True)
        
    model_ld.class_to_idx = checkpoint_ld['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(in_features=25088, out_features=4096)),
            ('relu1',nn.ReLU()),
            ('dropout1',nn.Dropout(p=0.5)),
            ('fc2',nn.Linear(in_features=4096, out_features=checkpoint_ld['hidden_units'])),
            ('relu2',nn.ReLU()),
            ('dropout2',nn.Dropout(p=0.5)),
            ('fc3',nn.Linear(in_features=checkpoint_ld['hidden_units'],out_features=102)),
            ('output',nn.LogSoftmax(dim=1))
        ]))

    model_ld.classifier = classifier
    
    for param in model_ld.parameters():
        param.requires_grad = False
    model_ld.load_state_dict(checkpoint_ld['state_dict'])
    return model_ld,checkpoint_ld['epochs'] , optimiz, criteria, scheduler
    
model_ld, eps_ld, op_ld, cr_ld, schd_ld = load_model_from_path(checkpoint)


#Converting Image(.jpg) to numpy array
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)

    #Resizing using thumbnail
    image.thumbnail((256,256), Image.ANTIALIAS)
    
    width, height = image.size
    crop_size = 224
    
    #Margins for crop
    left_margin = (width - crop_size)/2
    right_margin = left_margin + crop_size
    bottom_margin = (height - crop_size)/2
    top_margin = bottom_margin + crop_size
    
    #Cropping
    image = image.crop((left_margin,bottom_margin,right_margin,top_margin))
    
    image = np.array(image)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    #Normalize
    image = (image/255-mean)/std
    
    #Color Channel adjustment
    image = image.transpose(2,0,1)
    
    return image

def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    
    if gpu:
        model.cuda()
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        model.cpu()
        image = torch.from_numpy(image).type(torch.FloatTensor)
        
    image.unsqueeze_(0)
    # TODO: Implement the code to predict the class from an image file    
    with torch.no_grad():
        outcome = model.forward(image)
        probabilities = torch.exp(outcome)
        top_probs, top_labels = probabilities.topk(topk)
    
    if gpu:
        top_probs, top_labels = top_probs.cpu().numpy()[0], top_labels.cpu().numpy()[0]
    else:
        top_probs, top_labels = top_probs.numpy()[0], top_labels.numpy()[0]
    
    idx_to_class = {val:key for key,val in model_ld.class_to_idx.items()}
    
    top_labels = [idx_to_class[i] for i in top_labels]
    top_flowers = [cat_to_name[i] for i in top_labels]
    
    return top_probs, top_labels, top_flowers

top_probs, top_labels, top_flowers = predict(img_path, model_ld, use_gpu,top_k)
print(f'probs:{top_probs} \n top_labels:{top_labels} \n top_flowers:{top_flowers}')
