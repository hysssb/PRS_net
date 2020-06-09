import os
import nrrd
import torch
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms

class ShapeNetData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.istrain = (opt.phase == 'train')

        self.nameList = []
        self.size = 0
        self.transform = transforms.ToTensor()

        if self.istrain is True:
            fileName = 'train.csv'
        else:
            fileName = 'val.csv'
        with open(os.path.join(self.root, fileName)) as file:
            for f in file:
                self.nameList.append(f.strip("\n"))
                self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        print('index', index)
        nrrdPath = os.path.join(self.root, 'traindata', self.nameList[index], "model.nrrd")
        nrrdData, nrrd_options = nrrd.read(filename=nrrdPath)
        voxel = self.transform(nrrdData)
        voxel = voxel.view(1, 32, 32, 32)

        # read cloud points data
        #cloud_dir = os.path.join(self.root, 'cloud_data\\cloud\\')
        NewPcdPath = os.path.join(self.root, 'traindata\\', self.nameList[index], "model_new.pcd")

        with open(NewPcdPath, mode='r') as pcdFile:
            cloud_points = list()
            for i in range(1000):
                line = pcdFile.readline()
                # print('line', line)
                coord = line.strip('\n').split(' ')
                # print('coord0', coord[0])
                x = float(coord[0])
                y = float(coord[1])
                z = float(coord[2])
                # print(coord)
                cloud_points.append([x, y, z])
        cloud_points = torch.tensor(cloud_points)

        # read closest point of each voxel grid
        closeset = torch.zeros(1, 32, 32, 32)
        ClosestPath = os.path.join(self.root, 'traindata\\', self.nameList[index], "model.npv")
        with open(ClosestPath, mode='r') as npvFile:
            allrow = npvFile.readline()
            indices = allrow.strip('\n').split(' ')
            temp = 0
            for i in range(32):
                for j in range(32):
                    for k in range(32):
                        closeset[0, i, j, k] = int(indices[temp])
                        temp += 1

        sample = {
            'voxel': voxel,
            'points': cloud_points,
            'closeset': closeset
        }
        #print('ok for this sample')

        return sample
