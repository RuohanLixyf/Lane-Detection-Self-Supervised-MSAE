import argparse

# globel param
# dataset setting
img_width = 256 
img_height = 128
patch_size=16
mask_ratio=0.5
img_channel = 3
class_num_pretrain = 3
class_num_train = 2
label_width = 256
label_height = 128
label_channel = 1
data_loader_numworkers = 2#follow your device
#path
train_path = "./data/train_sample.txt"# same dataset as pretain
val_path = "./data/val_sample.txt"# same dataset as preval
test_path ="./data/test_index_0601all_desktop.txt"
save_path_train = "./save/train_results/"
save_path_pretrain = "./save/pretrain_results/"
save_path_test = "./save/test/"
pretrained_path='./model_pretrain/epoch1_valloss0.0039887395687401295_RAdam_0.001.pth'# pre-trained model path

# weight
class_weight = [0.02, 1.02]

def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch UNet_ConvLSTM')
    parser.add_argument('--model',type=str, default='UNet_ConvLSTM',help='( / UNet_ConvLSTM | SCNN_UNet_ConvLSTM | SCNN_UNet_Attention')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 69)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',#default=999
                        help='number of epochs to train (default: 99)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',#default=10
                        help='how many batches to wait before logging training status')
    parser.add_argument('--alpha', type=float, default=0.15, metavar='N',#default=
                        help='how many batches to wait before logging training status')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='N',#default=
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epsilon', type=float, default=0.1, metavar='N',#default=
                        help='how many batches to wait before logging training status')
    parser.add_argument('--loss', type=str, default='FocalLoss_poly', metavar='N',#default=FocalLoss
                        help=' / CrossEntropyLoss / FocalLoss_poly ')
    parser.add_argument('--optimizer', type=str, default='RAdam', metavar='N',#default=FocalLoss
                    help=' / RAdam / Adam / SGD  ')
    args = parser.parse_args()
    return args
