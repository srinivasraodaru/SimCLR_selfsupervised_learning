from torchvision.transforms import transforms
from torchvision import transforms, datasets
from torch import randperm
import numpy as np

np.random.seed(0)



class ZigZag(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        img = sample
        #####zigzag tranformer
        img = img.reshape(3,2,16,32).transpose(2,3).reshape(3,4,16,16)
        img = img[:,randperm(4),:,:]
        img = img.reshape(3,2,32,16).transpose(2,3).reshape(3,32,32)
        return img

class ContrastiveLearning:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def simclrTranform(size, s=1):
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
#                                              transforms.RandomHorizontalFlip(),
#                                              transforms.RandomApply([color_jitter], p=0.8),
#                                              transforms.RandomGrayscale(p=0.2),
#                                              transforms.ToTensor()])
        zigzag_transform = transforms.Compose([transforms.ToTensor(),
                                               ZigZag()])
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor(),
                                              ZigZag()])

        return data_transforms, zigzag_transform

    def dataSet(self, name):
        datasets_dict = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.simclrTranform(32)),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.simclrTranform(96)),
                                                          download=True),
                          'imagenet': lambda: datasets.ImageNet(self.root_folder, split='train',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.simclrTranform(32)),
                                                          download=True)}


        dataset_fn = datasets_dict[name]

        return dataset_fn()

class ContrastiveLearningViewGenerator(object):

    def __init__(self, base_transform):
        self.base_transform, self.zigzag_transform = base_transform


    def __call__(self, x):
        return [self.base_transform(x),self.zigzag_transform(x) ]
#        return [self.base_transform(x),self.base_transform(x)]
