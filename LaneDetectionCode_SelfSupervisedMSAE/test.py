import torch
import config
from config import args_setting
from dataset import RoadSequenceDatasetList
from model import generate_model
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from thop import profile, clever_format

args = args_setting()
data_name='normal'
model_name= args.model
test_path='./test_index/test_index_0601all_desktop.txt'#/normal dataset: test_index_0601all_desktop/ challenge dataset: test_index_stride1_use
save_path='./save/test/normal/'
pretrained_path = './model/UNet_ConvLSTM_97.65084160698785_lrRAd_batch0.001_epoch4UNet_ConvLSTM.pth'
file_name='./save/results.csv'

def output_result(model, test_loader, device):
    model.eval()
    k = 0
    feature_dic=[]
    with torch.no_grad():
        for sample_batched in test_loader:
            k+=1
            print(k)
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            output, feature = model(data)
            feature_dic.append(feature)
            pred = output.max(1, keepdim=True)[1]
            img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255 
            img = Image.fromarray(img.astype(np.uint8))
            data = torch.squeeze(data).cpu().numpy()
            data = np.transpose(data[-1], [1, 2, 0]) * 255
            data = Image.fromarray(data.astype(np.uint8))
            rows = img.size[0]
            cols = img.size[1]
            for i in range(0, rows):
                for j in range(0, cols):
                    img2 = (img.getpixel((i, j)))
                    if (img2[0] > 200 or img2[1] > 200 or img2[2] > 200):
                        data.putpixel((i, j), (234, 53, 57, 255))
            data = data.convert("RGB")
            data.save(save_path + "%s_data.jpg" % k)
            img.save(save_path + "%s_test.jpg" % k)
            
def evaluate_model(model, test_loader, device, criterion):
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
            output,feature = model(data) #
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

    precision = precision / (len(test_loader.dataset) - error)
    recall = recall / (len(test_loader.dataset) - error)
    F1_measure = F1_measure / (len(test_loader.dataset) - error)
    print('Precision: {:.5f}, Recall: {:.5f}, F1_measure: {:.5f}\n'.format(precision,recall,F1_measure))
    
    fileName = file_name
    number = 1
     
    books = []
    book = {
     'data_name': data_name,
     'model_name': model_name,
     'test_acc':test_acc, 'precision': precision, 'recall': recall, 'F1_measure': F1_measure
    }
    books.append(book)
    data = pd.DataFrame(books)
    csv_headers = ['data_name', 'model_name','test_acc', 'precision', 'recall', 'F1_measure']
    data.to_csv(fileName, header=csv_headers, index=False, mode='a+', encoding='utf-8')
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

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('parameter*4=filesize')
    return {'Total': total_num, 'Trainable': trainable_num}

from thop import profile
# args = args_setting()
# flops, params = profile(generate_model(args), inputs=(input,))
#print(flops)
#print(params)


if __name__ == '__main__':
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load data for batches, num_workers for multiprocess
    #if args.model == ('SegNet-ConvLSTM') or args.model == ('UNet-ConvLSTM'):

    test_loader=torch.utils.data.DataLoader(
        RoadSequenceDatasetList(file_path=test_path, transforms=op_tranforms),
        batch_size=1, shuffle=False, num_workers=0) #num_workers change to 0


    # load model and weights
    model = generate_model(args)    
    model.cuda() #Add for parallel Check if or not

    #model = nn.DataParallel(model)#Add for parallel
    
    class_weight = torch.Tensor(config.class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    
    pretrained_dict = torch.load(pretrained_path)
         
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)
    
    #output the result pictures
    output_result(model, test_loader, device)
    # calculate the values of accuracy, precision, recall, f1_measure
    print(pretrained_path)
    #get_parameter_number(model)
    evaluate_model(model, test_loader, device, criterion)
    print(get_parameter_number(model))
    
'''    #
    datasetsample = next(iter(test_loader))
    input = datasetsample['data']
    #model = generate_model(args)
    #model = model.to(device)
    flops, params = profile(model, inputs=(input.cuda(),)) # input.cuda()
    print({'flops': flops, 'params': params})
    
    #对于unet相关的两种方法计算的params不一样
    #print(params)
    #params = list(net.parameters())
    #print(len(params))
    #(params[0].size())  # conv1's .weight
'''

'''
sum(p.numel() for p in UNet_ConvLSTM_Mask().parameters())

model = generate_model(args)
model = model.to(device)
from modelsummary import summary
input = datasetsample['data']
inputs=(input.cuda())
summary(model, inputs, show_input=False)
summary(model, inputs, show_input=True)
#summary(UNet(3,2), torch.zeros((1, 3, 128, 256)), show_input=True) 

import torchsummary
torchsummary.summary(model, (3, 128, 256))

from torchsummaryX import summary
summary(model, inputs)

from torchsummary import summary
summary(model, (1, 28, 28))



import sys
class Logger(object):
    def __init__(self, filename='default3.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(stream=sys.stdout)




### Summary Use 
from torchsummary import summary
datasetsample = next(iter(test_loader))
input = datasetsample['data']
inputs=(input.cuda())
# summary(SCNN_UNet_ConvGRU(3,2), inputs)

summary(UNet_ConvLSTM_Mask(3,2), torch.zeros(1, 3, 3, 128, 256))
###
'''