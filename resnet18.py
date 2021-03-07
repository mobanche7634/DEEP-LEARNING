from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import time
import matplotlib.pyplot as plt

##resnet18##
class basicblock(nn.Module):
    expansion=1

    def __init__(self, input_channel, output_channel, strides=1, downsample=None):
        super(basicblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=output_channel, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conc2d(in_channels=input_channel,out_channels=output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu=nn.ReLU()
        self.downsample= downsample

    def forward(self, x):
        identity= x
        if self.downsample is not None:
            identity= self.downsample(x)
        y=self.relu(self.bn1(self.conv1(x)))
        y=(self.bn2(self.conv2(y)))
        y=self.relu(identity + y)
        return y


class Resnet(nn.module):
    def __init__(self,block,block_num, num_classes=2, include_top=True):
        super(Resnet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3,self.in_channel, kernel_size=7,stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inchannel)
        self.relu =nn.relu(inplace=True)
        self.maxpool= nn.Maxpool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=self._make_layer(block, 64, block_num[0])
        self.layer2= self._make_layer(block, 128, block_num[1],stride=2)
        self.layer3= self._make_layer(block, 256, block_num[2],stride=2)
        self.layer4= self._make_layer(block, 512, block_num[3],stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1)) #自适应平均池化层，输出特征矩阵的高和宽都是（1，1）
            self.fc = nn.Linear(512*block.expansion,num_classes)

        for m in self.moudles():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiminig_normal(m.weight, mode='fan_out',)


    def _make_layer(self, block, channel,block_num, stride=1):
        downsample= None
        downsample=nn.sequential(nn.Conv2d(self.in_channel, channel*block.expansion, kernel_size=1,stride=stride, bias=False),
                                 nn.BatchNorm2d(channel*block.expansion))
        
        layers = []
        layers.append(block(self.in_channel,channel,downsample=downsample,stride=stride))
        self.in_channel = channel*block.expansion

        for _ in range(1,block.num):
            layers.append(block(self.in_channel,channel))

        return nn.sequential(*layers)
        
    def forward(self,x):
        y=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        y=self.layer4(self.layer3(self.layer2((self.layer1(y)))))
        
        if self.include_top:
            y=self.avgpool(y)
            y=torch.flatten(y,1)
            y=self.fc(x)

        return y
def resnet18(num_classes=2,include_top=True):
    return resnet18(basicblock,[2,2,2,2],num_classes=num_classes,include_top=include_top)


###data processing,Data augmentation and normalization for training###
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
###totensor图片形状矩阵表示###
data_dir = 'pretraining'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
###shuffle每次刷新数据，workers工作进程数，batchsize每次迭代的样本数###




###进行了num_epoch遍样本，每遍batchsize个样本###
def traintest_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())  ###先深拷贝一份当前模型的参数，后面迭代过程中若遇到更优模型则替换。###
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to test mode

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
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
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

#display predictions for a few images#
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                ax.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
       
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###优先GPU，否则CPU###
#Load a pretrained model and reset final fully connected layer
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device) #调入GPU#
criterion = nn.CrossEntropyLoss() #交叉熵损失#
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs,减少学习率
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


#train and evaluate
model_ft = traintest_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

