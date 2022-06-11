import argparse
import torch
import torch.backends.cudnn as cudnn

from dataSetCreater import ContrastiveLearning
from models.resnetSimCLR import ResNetSimCLR
from SimCLR import SimCLR
from torchvision import models


parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='/content/drive/MyDrive/ece285_project/DataSets',
                    help='path to dataset directory')
parser.add_argument('-runsDir', metavar='RUNSDIR', default='/content/drive/MyDrive/ece285_project/other_projects/SimCLR_v1/Runs',
                    help='path to runs directory')
parser.add_argument('-run', metavar='RUNS_DIR', default='temp',
                    help='run name or type')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10', 'imagenet'])
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-bs', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--out_dim', default=10, type=int,
                    help='feature dimension')
parser.add_argument('--temperature', default=0.1, type=float,
                    help='softmax temperature')
                    
def main():
    args = parser.parse_args()
    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True

    ######dataset
    dataSet_SimCLR = ContrastiveLearning(args.data)
    trainDataSet = dataSet_SimCLR.dataSet(args.dataset_name)
    trainLoader = torch.utils.data.DataLoader(
            trainDataSet, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
            
            
            
    ####model and training
    model = ResNetSimCLR(args.out_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    simclrModel = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclrModel.train(train_loader)

if __name__ == "__main__":
    main()
