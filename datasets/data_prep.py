import torch
import torchvision.transforms as transforms
import nrrd
import random
import os

transform = transforms.ToTensor()
dataroot = 'D:\\python\\PRS_NET_master\\datasets\\'
# to get the model.nrrd files save in vector
for idx in range(1, 170, 1):
    VoxelPath = 'voxel_data\\voxel\\'
    voxel_dir = os.path.join(dataroot, VoxelPath)
    nrrdPath = os.path.join(voxel_dir, 'datarst', str(idx), "model.nrrd")
    nrrdData, nrrd_options = nrrd.read(filename=nrrdPath)
    voxel = transform(nrrdData)
    #print('voxel_1', voxel)
    voxel = voxel.view(1, 32, 32, 32)
    #print('voxel_2', voxel)

# to get the points as vectors saved in a list
    cloud_dir = os.path.join(dataroot, 'cloud_data\\cloud\\')
    pcdPath = os.path.join(cloud_dir, str(idx), "model.pcd")
    with open(pcdPath, mode='r') as pcdData:
        for temp in range(9):
            pcdData.readline()
        PointLine = pcdData.readline()
        num_points = PointLine.split()[1]
        num_points = int(num_points)
        # print('num_points: ', num_points)
        cloud_point = list()
        pcdData.readline()  # DATA ascii
        for xx in range(num_points):
            a = pcdData.readline()
            b = a.split()
            # b list !!!!ATTENTION: all are strs
            x = float(b[0])
            y = float(b[1])
            z = float(b[2])
            cloud_point.append([x, y, z])

    # add up to 1000 points
    #if 800 < num_points < 1000:
    #    add_points = random.sample(cloud_point, 1000-num_points)
    #    cloud_point.extend(add_points)
    #    num_points = 1000
    # use PU-Net to upsample maybe better
    if num_points < 1000:
        print('need to removed', idx)

    # align to voxel 2 find the close point
    cloud_point = torch.tensor(cloud_point)
    max, _ = torch.max(cloud_point, dim=0)
    min, _ = torch.min(cloud_point, dim=0)
    maxDx = torch.max(max - min)
    cloud_point = cloud_point - min
    cloud_point = cloud_point / maxDx * 32
    NewPcdPath = os.path.join(cloud_dir, str(idx), "model_new.pcd")
    with open(NewPcdPath, mode='w') as pcdFile:
        if num_points == 1000:
            for i in range(1000):
                pcdFile.write('{} {} {}\n'.format(float(cloud_point[i][0]),
                                                  float(cloud_point[i][1]),
                                                  float(cloud_point[i][2])))

    # find the closest set of cloud points
    nearestPointOfVoxel = torch.zeros(1, 32, 32, 32)
    #print(nearestPointOfVoxel)
    for i in range(32):
        for j in range(32):
            for k in range(32):
                voxelPos = torch.tensor([i, j, k])
                distance = torch.norm(cloud_point - voxelPos, dim=1, keepdim=True)
                _, index = torch.min(distance, dim=0)
                nearestPointOfVoxel[0, i, j, k] = index

    # 输出最近点文件
    npvPath = os.path.join(cloud_dir, str(idx), "model.npv")
    with open(npvPath, mode='w') as npvFile:
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    npvFile.write('{} '.format(
                        int(nearestPointOfVoxel[0, i, j, k])))
