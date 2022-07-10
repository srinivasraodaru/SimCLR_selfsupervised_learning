import logging
import os
import sys
import shutil
import torch
import torch.nn.functional as F
import yaml

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def accuracy(output, target):
    with torch.no_grad():
 
        batch_size = target.size(0)

        _, predictions = output.topk(1, 1, True, True)
        predictions = predictions.t()
        correct = predictions.eq(target.view(1, -1).expand_as(pred))
        res = correct[:1].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
        return res

def saveFile(modelFolder, args):
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)
        with open(os.path.join(modelFolder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)



class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        logging_folder = os.path.join(self.args.runsDir, self.args.run)
        if not os.path.exists(logging_folder):
                os.makedirs(logging_folder)
                
        self.writer = SummaryWriter(log_dir = logging_folder)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        
        
    def train(self, trainLoader):
        ##save config file
        saveFile(self.writer.log_dir, self.args)
        
        for epoch in range(self.args.epochs):
            for images, label in tqdm(train_loader):
            
                ###forward
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                representations = self.model(images)
                logits, labels = self.nceLoss(representations)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()

                acc = accuracy(logits, labels)
                self.writer.add_scalar('loss', loss, global_step=epoch)
                self.writer.add_scalar('acc', acc, global_step=epoch)

        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        checkpoint_folder = os.path.join(self.args.runsDir, self.args.run)
        if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
    
        torch.save(self.model.state_dict(),  filename=os.path.join(checkpoint_folder, checkpoint_name))

        def nceLoss(delf, features):
        
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        ###find the simiilarity matrix to be used in NCE loss
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels
