import torchvision.datasets as dset
import torch 
import os
import torch.nn as nn
import torch.optim as optim
import model
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

def trainmodel(net, criterion,optimizer,epochs,train_data_loader,device,
               scheduler,train_num):         
    net.train()
    running_loss = 0.0
    for step in enumerate(train_data_loader):
            
            optimizer.zero_grad()           
            batch_input, batch_label=next(iter(train_data_loader))
            inputs=batch_input.to(device)
            labels=batch_label.to(device)
            
            logits = net(inputs)
            _, preds = torch.max(logits, 1)
            loss = criterion(logits, labels)
            loss.backward() 
            optimizer.step()
            running_loss += loss.item()       
         
    scheduler.step()
    train_loss= running_loss/train_num

    return train_loss

def valmodel(net,criterion,epochs,val_data_loader,device, save_path,best_acc,val_num):    
    net.eval()
    running_corrects = 0
    with torch.no_grad():
            
            for i in val_data_loader:            
                batch_input, batch_label= i
                inputs=batch_input.to(device)
                labels=batch_label.to(device)
                logits = net(inputs)                
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels).item()
                val_loss= loss/val_num
                running_corrects += torch.eq(preds, labels).sum().float().item()
            epoch_acc = running_corrects / val_num              

    if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(net.state_dict(), save_path)
                        
    return val_loss,epoch_acc,best_acc,save_path

def testmodel(net,criterion,epochs,test_data_loader,device, save_path,best_acc,test_num):    
    net.eval()
    running_corrects = 0
    with torch.no_grad():           
            for i in test_data_loader:            
                batch_input, batch_label= i
                inputs=batch_input.to(device)
                labels=batch_label.to(device)
                logits = net(inputs)                
                _, preds = torch.max(logits, 1)
                running_corrects += torch.eq(preds, labels).sum().float().item()
            epoch_acc = running_corrects / test_num              

    if epoch_acc > best_acc:
            best_acc = epoch_acc
                        
    return best_acc

def model_conv(save_path,device):
    net_conv =model.resnet18()
    in_channel = net_conv.fc.in_features
    net_conv.fc = nn.Linear(in_channel, 2)
    net_conv.load_state_dict(torch.load(save_path))
    
    for param in net_conv.parameters():
        param.requires_grad = False
    
    in_channel = net_conv.fc.in_features
    net_conv.fc = nn.Linear(in_channel, 2)    
    net_conv.to(device)    
    # define loss function
    criterion = nn.CrossEntropyLoss()
    # construct an optimizer
    #params = [p for p in net.parameters() if p.requires_grad]
    optimizer_conv = optim.SGD(net_conv.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=0.1)
    
    return net_conv,criterion,optimizer_conv,scheduler

def get_data(k, i, redata,batch_sizes):  
    fold_size = len(redata) // k 
    start = fold_size
    end = 2*fold_size
    if i == 0:
        train_data =redata[0:start]
        val_data = redata[start:end]        
        test_data = redata[end:]
    if i == 1:
        train_data =redata[start:end]
        val_data = redata[end:]        
        test_data = redata[0:start]   

    else:  
        train_data =redata[end:]
        val_data = redata[0:start]
        test_data = redata[start:end]
        
    num_train = len(train_data)
    num_val = len(val_data)
    num_test = len(test_data)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sizes,shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_sizes,shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_sizes,shuffle=False)
    
    return train_data_loader,val_data_loader,test_data_loader,num_train,num_val,num_test
    
def main(data_road):
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    batch_sizes = 100
    k=3
    nw = 4  # number of workers
    print('Using {} dataloader workers every process'.format(nw))   
    
    
    transform = transforms.Compose([transforms.Resize(225),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])]) 
    data = dset.ImageFolder(data_road,transform=transform)
    redata=tuple()
    print(len(data))
    for i in range(len(data)//2):
        redata = redata+(data[i],data[-1-i])
    print(len(redata[0:3]))
    
    
          
          
    net = model.resnet18()  
    #https://download.pytorch.org/models/resnet18-5c106cde.pth
    model_weight_path = "./resnet18-5c106cde.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    #finetuning
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    net.to(device)
    
    # define loss function
    criterion = nn.CrossEntropyLoss()
    # construct an optimizer
    optimizer= optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    epochs = 3
    train_loss=0
    val_loss=0    

    save_path = './resNet34.pth'
    best_acc=0
    epoch_acc=0
    sum = 0
    for i in range(k):
        train_data_loader,val_data_loader,test_data_loader,train_num,val_num,test_num = get_data(k,i,redata,batch_sizes)
        #for epoch in range(epochs):
        #   print('epoch[{}/{}]'.format(epoch+1,epochs))
        #   train_loss = trainmodel(net,criterion,optimizer,epochs,train_data_loader,device,scheduler,train_num)
        #   val_loss, epoch_acc,best_acc,save_path=valmodel(net,criterion,epochs,val_data_loader,device,save_path,best_acc,val_num)
        #  print("Train loss:",train_loss)
        # print("Val loss:", val_loss,"Val Acc:",epoch_acc)  
     # print("Best Acc:",best_acc)
     # print('finish')

        #freezing all layers exceot final layer
        net_conv,criterion,optimizer,scheduler=model_conv(save_path,device)
        print("************Training again**************")
        for epoch in range(epochs):
            print('epoch[{}/{}]'.format(epoch+1,epochs))
            train_loss = trainmodel(net_conv,criterion,optimizer,epochs,train_data_loader,device,scheduler,train_num)
            val_loss, epoch_acc,best_acc,save_path=valmodel(net_conv,criterion,epochs,val_data_loader,device,save_path,best_acc,val_num)
            print("Train loss:",train_loss)
            print("Val loss:", val_loss,"Val Acc:",epoch_acc)
        print("Find the best validation accurcy and store the weights")
        best_acc=testmodel(net,criterion,epochs,test_data_loader,device, save_path,best_acc,test_num)
        
        
        sum += best_acc
    mean_acc = sum/3
    print('********Mean Acc:',mean_acc)
    print('finish training')
    return mean_acc

    

if __name__ == '__main__':
    data_road = 'data/catdog'
    main(data_road)