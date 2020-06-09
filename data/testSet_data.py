# dataset for test

import os
import nrrd
import torch
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms

class testSetData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot

        self.nameList = []
        self.size = 0
        self.transform = transforms.ToTensor()

        fileName = 'var.csv'
        with open(os.path.join(self.root, fileName)) as file:
            for f in file:
                self.nameList.append(f.strip("\n"))
                self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        # read voxel data
        VoxelPath = 'test_data\\testdata\\'
        voxel_dir = os.path.join(self.root, VoxelPath)
        nrrdPath = os.path.join(voxel_dir, self.nameList[index], "model.nrrd")
        nrrdData, nrrd_options = nrrd.read(filename=nrrdPath)
        voxel = self.transform(nrrdData)
        voxel = voxel.view(1, 32, 32, 32)

        # read cloud points data
        NewPcdPath = os.path.join(voxel_dir, self.nameList[index], "model_new.pcd")
        with open(NewPcdPath, mode='r') as pcdFile:
            cloud_points = list()
            for i in range(1000):
                line = pcdFile.readline()
                coord = line.split(' ')
                cloud_points.append([float(coord[0]), float(coord[1]), float(coord[2])])
        cloud_points = torch.tensor(cloud_points)

        # read closest point of each voxel grid
        closeset = torch.zeros(1, 32, 32, 32)
        ClosestPath = os.path.join(voxel_dir, self.nameList[index], "model.npv")
        with open(ClosestPath, mode='r') as npvFile:
            allrow = npvFile.readline()
            indices = allrow.strip('\n').split(' ')
            temp = 0
            for i in range(32):
                for j in range(32):
                    for k in range(32):
                        closeset[0, i, j, k] = int(indices[temp])
                        temp += 1

        sample_test = {
            'voxel': voxel,
            'points': cloud_points,
            'closeset': closeset
        }

        return sample_test
