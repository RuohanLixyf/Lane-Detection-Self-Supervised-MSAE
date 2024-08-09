import os
import torch
import torch.nn as nn
import config
import time
from config import args_setting
from dataset import RoadSequenceDatasetList_MASK
from pretrain_model import generate_model
from torchvision import transforms
from torch.optim import lr_scheduler
from torchvision.models import vgg11
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import FocalLoss_poly
from radam import RAdam
import cv2
import pandas as pd

def train(args, epoch, model, train_loader, device, optimizer, criterion):

    since = time.time()
    model.train()
    for batch_idx,  sample_batched in enumerate(train_loader):
        data= sample_batched['data'].to(device)  
        target=sample_batched['label'].to(device)
        optimizer.zero_grad()
        output, aux = model(data)
        output=output.to(torch.float32)
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

def val(args, model, val_loader, device, criterion, best_acc, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample_batched in val_loader:
            data= sample_batched['data'].to(device)  
            target=sample_batched['label'].to(device)
            output, aux = model(data) # if mpt unt-convlestm then output=            output= model(data)
            output=output.to(torch.float32)
            test_loss += criterion(output, target)

    test_loss /= (len(val_loader.dataset)/args.test_batch_size)
    print('\nAverage loss: {:.4f}\n'.format(test_loss))
    torch.save(model.state_dict(), './model_pretrain/epoch{}_valloss{}_{}_{}.pth'.format(epoch,test_loss,args.optimizer,args.lr))
    
    #show and save results
    img = torch.squeeze(output).cpu().numpy()    
    img=np.transpose(img,[1, 2, 0]) * 255    
    img = Image.fromarray(img.astype(np.uint8))    
    to_PILimage = transforms.ToPILImage()
    data0 = torch.squeeze(data).cpu().numpy()
    data0 = np.transpose(data0[-1], [1, 2, 0])* 255
    data0 = Image.fromarray(data0.astype(np.uint8))
    rows = img.size[0]
    cols = img.size[1]
    data0 = data0.convert("RGB")
    img = img.convert("RGB")    
    data0.save(config.save_path_pretrain + "%s_data_pretrain.jpg" % epoch)#red line on the original image
    img.save(config.save_path_pretrain + "%s_pred_pretrain.jpg" % epoch)#prediction result
    data0.show()    
    img.show()  
    target = torch.squeeze(target).cpu().numpy()
    target0 = np.transpose(target,[1, 2, 0]) * 255
    target0 = Image.fromarray(target0.astype(np.uint8))
    target0 = target0.convert("RGB")
    target0.show()
    target0.save(config.save_path_pretrain + "%s_target_pretrain.jpg" % epoch)
    
    #write the parameter to file
    fileName = './save/pretrain_results.csv'
    number = 1
    books = []
    book = {
      'num': number,
      'optimizer': args.optimizer,
      'batch size': args.batch_size,
      'epoch': epoch,
      'mask_ratio': config.mask_ratio,
      'mask_num': 5,
      'Average loss':test_loss
     
    }
    books.append(book)
     
    data = pd.DataFrame(books)
    if epoch == 1:
        csv_headers = ['num', 'optimizer','batch size','epoch', 'mask_ratio', 'mask_num', 'Average loss']
        data.to_csv(fileName, header=csv_headers, index=False, mode='a+', encoding='utf-8')
    else:
        data.to_csv(fileName, header=False, index=False, mode='a+', encoding='utf-8')
        number = number + 1

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


if __name__ == '__main__':
    print(os.getcwd())
    args = args_setting()
    #torch.multiprocessing.set_start_method('spawn')
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda' if use_cuda else "cpu") #Note This
    torch.backends.cudnn.benchmark = True
    
    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])
    # load data for batches, num_workers for multiprocess
    print('mode:',args.model)
    print('Pre-training...')
    #dataloader
    train_loader = torch.utils.data.DataLoader(
        RoadSequenceDatasetList_MASK(file_path=config.train_path, transforms=op_tranforms),
        batch_size=args.batch_size,shuffle=True,num_workers=config.data_loader_numworkers)
    val_loader = torch.utils.data.DataLoader(
        RoadSequenceDatasetList_MASK(file_path=config.val_path, transforms=op_tranforms),
        batch_size=args.test_batch_size,shuffle=True,num_workers=config.data_loader_numworkers)
    test_loader=torch.utils.data.DataLoader(
        RoadSequenceDatasetList_MASK(file_path=config.test_path, transforms=op_tranforms),
        batch_size=1, shuffle=False, num_workers=config.data_loader_numworkers) #num_workers change to 0


    #load model
    model = generate_model(args)
    model.cuda() #Add for parallel
    model = nn.DataParallel(model) #Add for parallel

    optimizer = RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999)) #change optimizer note
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    #MSELoss
    criterion = torch.nn.MSELoss()
    
    best_acc = 0
    best_F1 = 0 


    for epoch in range(1, args.epochs+1):
        #scheduler.step()
        train(args, epoch, model, train_loader, device, optimizer, criterion)
        scheduler.step()
        val(args, model, val_loader, device, criterion, best_acc, epoch)
        #scheduler.step()
        print('\n lr:')
        print(scheduler.get_last_lr())
