import matplotlib.pyplot as plt
import time
import copy
from collections import OrderedDict
from PIL import Image
import numpy as np
import seaborn as sns
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('data_dir',help="root data directory")
parser.add_argument('--arch', required=False, type=str, default='vgg16', help="Choose Pre-trained models available for transfer learning; Default is vgg16; Other available model vgg13")
parser.add_argument('--save_dir', required=True, help="checkpoint path to save model")
parser.add_argument('--learning_rate', required=False,type=float, default=0.01,help="learning rate for the model; Default is 0.01")
parser.add_argument('--hidden_units', required=False, type=int, default=1024, help="Return top KK most likely classes; Default is 1024")
parser.add_argument('--epochs', required=False, type=int, default=10, help="No.of times to iterate through model; Default is 10")
parser.add_argument('--gpu', required=False, action="store_true",help="Use GPU to compute")

args = parser.parse_args()


# ## Load the data

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

device = 'cuda' if args.gpu else 'cpu'

# Defining your transforms for the training, validation, and testing sets
data_transforms = {'train':transforms.Compose([transforms.RandomRotation(45),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])                                      
                                     ]),
                   'validation_test':transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])                                      
                                     ])}


# Loading the datasets with ImageFolder
image_datasets = {'train':datasets.ImageFolder(train_dir,data_transforms['train']),
                 'valid':datasets.ImageFolder(valid_dir,data_transforms['validation_test']),   
                 'test':datasets.ImageFolder(test_dir,data_transforms['validation_test'])}
                    
# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {i:torch.utils.data.DataLoader(image_datasets[i],batch_size=32,shuffle=True) 
              for i in image_datasets}

dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}

# Building and training the classifier

if args.arch=='vgg16':
    model = models.vgg16(pretrained=True)
    print('model: vgg16')
elif args.arch=='vgg13':
    model = models.vgg13(pretrained=True)
    print('model: vgg13')



# Defining new classifier for our Model
classifier = nn.Sequential(OrderedDict([
    ('fc1',nn.Linear(in_features=25088, out_features=4096)),
    ('relu1',nn.ReLU()),
    ('dropout1',nn.Dropout(p=0.5)),
    ('fc2',nn.Linear(in_features=4096, out_features=args.hidden_units)),
    ('relu2',nn.ReLU()),
    ('dropout2',nn.Dropout(p=0.5)),
    ('fc3',nn.Linear(in_features=args.hidden_units,out_features=102)),
    ('output',nn.LogSoftmax(dim=1))
]))

# To freeze features network static
for param in model.parameters():
    param.requires_grad = False

# Setting a classifier to model
model.classifier = classifier

def train_model(model, criterion, optimizer, scheduler, num_epochs=25,device='cuda'):
    if device=='cuda':
        model.cuda()
    else:
        model.cpu()
        
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                print('Training...')
                model.train()  # Set model to training mode
            else:
                print('Validating...')
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


criteria = nn.NLLLoss()
optimA = optim.Adam(list(model.classifier.parameters()), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimA,step_size=4)

model_trained2 = train_model(model, criteria, optimA, scheduler, args.epochs, device)

def calc_accuracy(model1,data_set):
    model1.eval()
    model1.to(device='cuda')

    accuracy_rate = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data_set]):
            inputs, labels = inputs.cuda(), labels.cuda()

            #Output for test data from trained model
            output = model1.forward(inputs)

            #maximum probaility, predicted value
            max_log, predicted = output.max(dim=1)

            how_many_we_got_correct = predicted == labels.data
            accuracy_rate.append(how_many_we_got_correct.float().mean().cpu().numpy())

            if idx%5==0:
                print(f'Accuracy for batch {idx}:',accuracy_rate[-1])
    print(f'Overall accuracy on test data is {sum(accuracy_rate)*100/len(accuracy_rate)}')
    
calc_accuracy(model_trained2,'test')

model_trained2.cpu()
model_trained2.class_to_idx = image_datasets['train'].class_to_idx

torch.save({'state_dict':model_trained2.state_dict(),
            'optimizer_state':optimA.state_dict(),
            'scheduler':scheduler.state_dict(),
            'criteria':criteria.state_dict(),
            'arch':args.arch,
            'epochs':args.epochs,
            'hidden_units':args.hidden_units,
            'learning_rate':args.learning_rate,
            'class_to_idx':model_trained2.class_to_idx},
           args.save_dir)