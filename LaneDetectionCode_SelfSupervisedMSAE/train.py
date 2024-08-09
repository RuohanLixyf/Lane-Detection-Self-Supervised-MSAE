import torch
import config
import time
from config import args_setting
from dataset import RoadSequenceDatasetList
import model as model
from model import generate_model
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.nn as nn
import os
#from radam import RAdam
#import UNet_TwoConvGRU
import numpy as np
import cv2
import pandas as pd
from utils import FocalLoss_poly
from radam import RAdam
from PIL import Image
import torch.nn.functional as F

def train(args, epoch, model, train_loader, device, optimizer, criterion):
    since = time.time()
    model.train()

    for batch_idx,  sample_batched in enumerate(train_loader):
        data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        output, aux = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))

    time_elapsed = time.time() - since
    print('Train Epoch: {} complete in {:.0f}m {:.0f}s'.format(epoch,
        time_elapsed // 60, time_elapsed % 60))

def val(args, model, val_loader, device, criterion, best_acc,epoch):
    print('val..')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample_batched in val_loader:
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            output , aux = model(data) # output,_ = model(data) for Unetlstm change to output = model(data) for gegnet
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= (len(val_loader.dataset)/args.test_batch_size)
    val_acc = 100. * int(correct) / (len(val_loader.dataset) * config.label_height * config.label_width)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
        test_loss, int(correct), len(val_loader.dataset), val_acc))
    #torch.save(model.state_dict(), './model/%s.pth'%val_acc) #/scratch/ '"$TMPDIR"/LaneDetectionCode/save/%s.pth'
    #torch.save(model.state_dict(), './model/{}/{}_{}_lr{}_batch{}_epoch{}_{}_alpha{}_gamma{}_{}.pth'.format(args.model,val_acc,str(optimizer)[0:3],args.lr,args.batch_size,epoch,args.loss,args.alpha,args.gamma,args.model))


def get_parameters(model, layer_name):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.UpsamplingBilinear2d
    )
    for name, module in model.named_children():
        if name in layer_name:
            for layer in module.children():
                if isinstance(layer, modules_skipped):
                    continue
                else:
                    for parma in layer.parameters():
                        yield parma

def evaluate_model(model, test_loader, device, criterion, epoch):
    print('evaluate..')
    model.eval()
    i = 0
    precision = 0.0
    recall = 0.0
    test_loss = 0
    correct = 0
    error=0
    with torch.no_grad():
        for sample_batched in test_loader:
            i+=1
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            output , aux = model(data) #
            pred = output.max(1, keepdim=True)[1] 
            img = torch.squeeze(pred).cpu().numpy()*255
            lab = torch.squeeze(target).cpu().numpy()*255
            img = img.astype(np.uint8)
            lab = lab.astype(np.uint8)
            kernel = np.uint8(np.ones((3, 3)))
            #accuracy
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            #precision,recall,f1
            label_precision = cv2.dilate(lab, kernel)
            pred_recall = cv2.dilate(img, kernel)
            img = img.astype(np.int32)
            lab = lab.astype(np.int32)
            label_precision = label_precision.astype(np.int32)
            pred_recall = pred_recall.astype(np.int32)
            a = len(np.nonzero(img*label_precision)[1])
            b = len(np.nonzero(img)[1])
            if b==0:
                error=error+1
                continue
            else:
                precision += float(a/b)
            c = len(np.nonzero(pred_recall*lab)[1])
            d = len(np.nonzero(lab)[1])
            if d==0:
                error = error + 1
                continue
            else:
                recall += float(c / d)
            F1_measure=(2*precision*recall)/(precision+recall)

    test_loss /= (len(test_loader.dataset) / args.test_batch_size)
    test_acc = 100. * int(correct) / (len(test_loader.dataset) * config.label_height * config.label_width)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)'.format(
        test_loss, int(correct), len(test_loader.dataset), test_acc))
    #torch.save(model.state_dict(), './model/%s.pth'%test_acc) #/scratch/ '"$TMPDIR"/LaneDetectionCode/save/%s.pth'
    torch.save(model.state_dict(), './model/{}_{}_lr{}_batch{}_epoch{}UNet_ConvLSTM.pth'.format(args.model,test_acc,str(optimizer)[0:3],args.lr,args.batch_size,epoch))


    precision = precision / (len(test_loader.dataset) - error)
    recall = recall / (len(test_loader.dataset) - error)
    F1_measure = F1_measure / (len(test_loader.dataset) - error)
    print('Precision: {:.5f}, Recall: {:.5f}, F1_measure: {:.5f}\n'.format(precision,recall,F1_measure))
    evaluate_result = {'precision': precision, 'recall': recall, 'F1_measure': F1_measure, 'test_acc':test_acc}
    
    #save picture to file
    img1 = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255
    img1 = Image.fromarray(img1.astype(np.uint8))
    data0 = torch.squeeze(data).cpu().numpy()
    data0 = np.transpose(data0[-1], [1, 2, 0]) * 255
    data0 = Image.fromarray(data0.astype(np.uint8))
    rows = img1.size[0]
    cols = img1.size[1]
    for i in range(0, rows):
        for j in range(0, cols):
            img2 = (img1.getpixel((i, j)))
            if (img2[0] > 200 or img2[1] > 200 or img2[2] > 200):
                data0.putpixel((i, j), (234, 53, 57, 255))
    data0 = data0.convert("RGB")
    data0.show()
    img1.show()
    data0.save(config.save_path_train + "%s_data.jpg" % epoch)#red line on the original image
    img1.save(config.save_path_train + "%s_pred.jpg" % epoch)#prediction result

    return evaluate_result

#from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    #writer = SummaryWriter()
    print(os.getcwd())
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.is_available())

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load data for batches, num_workers for multiprocess
    train_loader = torch.utils.data.DataLoader(
        RoadSequenceDatasetList(file_path=config.train_path, transforms=op_tranforms),
        batch_size=args.batch_size,shuffle=True,num_workers=config.data_loader_numworkers)
    val_loader = torch.utils.data.DataLoader(
        RoadSequenceDatasetList(file_path=config.val_path, transforms=op_tranforms),
        batch_size=args.test_batch_size,shuffle=True,num_workers=config.data_loader_numworkers)
    test_loader=torch.utils.data.DataLoader(
        RoadSequenceDatasetList(file_path=config.test_path, transforms=op_tranforms),
        batch_size=1, shuffle=False, num_workers=config.data_loader_numworkers) #num_workers change to 0


    #load model
    model = generate_model(args)
    model.cuda() #Add for parallel
    # model = nn.DataParallel(model) #Add for parallel
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999)) #change optimizer note

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    if args.loss=='CrossEntropyLoss':
        class_weight = torch.Tensor(config.class_weight)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device) 
    elif args.loss=='FocalLoss_poly':
        criterion = FocalLoss_poly(alpha=args.alpha,gamma=args.gamma,epsilon=args.epsilon,size_average=True).to(device)

    best_acc = 0
    
    pretrained_dict = torch.load(config.pretrained_path)  #[]#add map_location ='cpu'  , map_location ='cpu'
    model_dict = model.state_dict()

    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1) #
    model.load_state_dict(model_dict)
    
    # train
    for epoch in range(1, args.epochs+1):
        print('lr---------', scheduler.get_last_lr())
        train(args, epoch, model, train_loader, device, optimizer, criterion)
        if scheduler.get_lr()[0] > 0.000000001:#0.0000001
            scheduler.step()
        else:
            print('lr----no--change--------')
        print('lr---------', scheduler.get_last_lr())
        val(args, model, val_loader, device, criterion, best_acc,epoch)
        result = evaluate_model(model, test_loader, device, criterion, epoch)

        if result['F1_measure'] > best_acc:
            best_acc = result['F1_measure']
            best_name='__test_acc=%s'%result['test_acc']  + '__precision=%s'%result['precision']  + '__recall=%s'%result['recall']  + '__F1_measure=%s'%result['F1_measure'] + '_epoch=%s'%epoch + '_'
            print('best testing-------------', best_name)
            print('test acc-------------',  result['test_acc'])
            print('precision-----------', result['precision'])
            print('recall-----------', result['recall'])
            print('F1_measure-----------', result['F1_measure'])
        elif result['F1_measure'] > 0.969:
            current_name='__test_acc=%s'%result['test_acc']  + '__precision=%s'%result['precision']  + '__recall=%s'%result['recall']  + '__F1_measure=%s'%result['F1_measure'] + '_epoch=%s'%epoch + '_'
            print('current testing evaluation F1 beat 0.969-------------', current_name)

         
