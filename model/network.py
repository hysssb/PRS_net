import torch
import torch.nn as nn
from options.train_options import TrainOptions
import util.Quaternion as Qtn
from torch.optim import lr_scheduler
import math

###############################################################################
# Helper Functions
###############################################################################
# set GPU


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_net(net, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    return net


def define_prs(gpu_ids):
    net = PRS_Net()
    return init_net(net, gpu_ids)


def define_loss_1(opt):
    loss_sd = LossSymmetryDistance()
    # loss_sd_ = torch.sum(loss_sd) / opt.batch_size
    return loss_sd


def define_loss_2(opt):
    loss_reg = LossRegularization()
    # loss_reg_ = torch.sum(loss_reg) / opt.batch_size
    return loss_reg




###############################################################################
# main pipeline, aligned to figure 1 in paper
###############################################################################
class PRS_Net(nn.Module):
    def __init__(self):
        super(PRS_Net, self).__init__()  # call father constructed function
        self.opt = TrainOptions().parse()
        print(self.opt)
        self.leaky_negSlope = self.opt.leaky_negSlope
        self.myLeakyReLU = nn.LeakyReLU(self.leaky_negSlope)
        self.myLayer1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.myLeakyReLU)
        self.myLayer2 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.myLeakyReLU)
        self.myLayer3 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.myLeakyReLU)
        self.myLayer4 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.myLeakyReLU)
        self.myLayer5 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.myLeakyReLU)
        self.myLayers = nn.Sequential(
            self.myLayer1,
            self.myLayer2,
            self.myLayer3,
            self.myLayer4,
            self.myLayer5
        )

        self.fcLayers = nn.Sequential(
            nn.Linear(64, 32),
            self.myLeakyReLU,
            nn.Linear(32, 16),
            self.myLeakyReLU,
            nn.Linear(16, 4),
            self.myLeakyReLU
        )

    def forward(self, voxel):
        print('voxel1', voxel.shape)
        self.outputs = torch.zeros(voxel.shape[0], 6, 4)  # (4,6,4) batchsize
        # convolution Layers & pool Layers extract the global features of the shape
        voxel = self.myLayers(voxel)
        voxel = voxel.view(voxel.shape[0], 64)
        print('voxel shape', voxel.shape)
        input0 = self.fcLayers(voxel)  # wrong!!!!
        print('input0', input0.shape)
        output0 = input0 / torch.norm(input0, dim=1)
        self.in2output(output0, 0)

        input1 = self.fcLayers(voxel)
        output1 = input1 / torch.norm(input1, dim=1)
        self.in2output(output1, 1)

        input2 = self.fcLayers(voxel)
        output2 = input2 / torch.norm(input2, dim=1)
        self.in2output(output2, 2)

        input3 = self.fcLayers(voxel)
        output3 = input3 / torch.norm(input3, dim=1)
        self.in2output(output3, 3)

        input4 = self.fcLayers(voxel)
        output4 = input4 / torch.norm(input4, dim=1)
        self.in2output(output4, 4)

        input5 = self.fcLayers(voxel)
        output5 = input5 / torch.norm(input1, dim=1)
        self.in2output(output5, 5)

        return self.outputs

    def in2output(self, x: torch.tensor, index: int):
        for i in range(self.outputs.shape[0]):
            self.outputs[i][index] = x[i]
        return

    def __call__(self, voxel):
        # batchsize*1*32*32*32
        return self.forward(voxel)


# to promote planar symmetry
# chapter 4.1
class LossSymmetryDistance(object):
    def __call__(self, outputs: torch.tensor, sample):
        self.loss = torch.zeros(outputs.shape[0], 6)
        for i in range(outputs.shape[0]):
            self.voxel = sample['voxel'][i]
            self.points = sample['points'][i]
            self.nstvoxel = sample['closeset'][i]
            self.output = outputs[i]  # 6*4 ith voxel

            for j in range(0, 3):
                plane = self.output[j]
                self.ProcessedPoints = self.symmTransform(plane)  # plane
                self.loss[i][j] = self.sumAllDistance()

            for j in range(3, 6):
                rotate = self.output[j]
                self.ProcessedPoints = self.rotateTransform(rotate)  # axis
                self.loss[i][j] = self.sumAllDistance()

        return self.loss

    def sumAllDistance(self):
        Distance4all = 0
        for i in range(self.points.shape[0]):
            x = int(self.ProcessedPoints[i][0])
            y = int(self.ProcessedPoints[i][1])
            z = int(self.ProcessedPoints[i][2])
            x = 31 if x > 31 else x
            y = 31 if y > 31 else y
            z = 31 if z > 31 else z
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            z = 0 if z < 0 else z
            index = int(self.nstvoxel[0, x, y, z])
            # p2 = self.points[index]
            # p1 = self.ProcessedPoints[i]
            d = torch.norm(self.points[index] - self.ProcessedPoints[i])
            d = d/1000
            # d = math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2) + math.pow((p2[2] - p1[2]), 2))
            Distance4all += d
        return Distance4all

    def symmTransform(self, plane: torch.tensor):
        outPoints = torch.zeros_like(self.points)  # 1000*3
        for pt in range(self.points.shape[0]):
            outPoints[pt] = self.points[pt] - \
                            plane[0:3] * (torch.dot(self.points[pt], plane[0:3]) + plane[3]) /\
                            math.sqrt(torch.dot(plane[0:3], plane[0:3]))
        return outPoints

    def rotateTransform(self, rotate: torch.tensor):
        outPoints = torch.zeros_like(self.points)
        for pt in range(self.points.shape[0]):
            outPoints[pt] = Qtn.rotate(rotate, self.points[pt])
        return outPoints

# to avoid producing duplicated symmetry planes
# chapter 4.2
class LossRegularization(object):
    def __call__(self, outputs: torch.tensor):
        self.loss = torch.zeros(outputs.shape[0])  # tensor([0., 0., 0., 0.])
        for i in range(outputs.shape[0]):
            M1 = outputs[i][0:3, 0:3]
            M2 = outputs[i][3:6, 1:4]
            # print('M1, M2', M1, M2)
            M1 = M1 / torch.norm(M1, dim=1).view(-1, 1)
            M2 = M2 / torch.norm(M2, dim=1).view(-1, 1)
            diagone = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            A = torch.mm(M1, M1.t()) - diagone
            B = torch.mm(M2, M2.t()) - diagone
            # print('A', A)
            # print('B', B)
            self.loss[i] = torch.sum(A**2) + torch.sum(B**2)

            # print('loss', i, self.loss[i])
        # print('loss', self.loss)
        return self.loss/1000

# to remove duplicated outputs: if its dihedral angle is less than π/6
# symmetry planes/rotation axes lead to high symmetry distance loss greater than 4 × 10−4
# figure 3 shows the examples
# chapter 5
class validateOutputs(object):
    def __call__(self, outputs: torch.tensor, lsd: torch.tensor, ml: float,
                 mc: float):
        self.isRemoved = [False, False, False, False, False, False]
        for i in range(6):
            if lsd[i] > ml:
                self.isRemoved[i] = True
        for i in range(2):
            if self.isRemoved[i] is True:
                continue
            for j in range(i + 1, 3):
                if self.isRemoved[j] is True:
                    continue
                if self.cosDihedralAngle(outputs[i][0:3],
                                         outputs[j][0:3]) > mc:
                    if lsd[i] > lsd[j]:
                        self.isRemoved[i] = True
                    else:
                        self.isRemoved[j] = True

        for i in range(3, 5, 1):
            if self.isRemoved[i] is True:
                continue
            for j in range(i + 1, 6):
                if self.isRemoved[j] is True:
                    continue
                if self.cosDihedralAngle(outputs[i][1:4],
                                         outputs[j][1:4]) > mc:
                    if lsd[i] > lsd[j]:
                        self.isRemoved[i] = True
                    else:
                        self.isRemoved[j] = True

        for i in range(6):
            if self.isRemoved[i] is True:
                outputs[i] = torch.zeros(4)
        return outputs

    def cosDihedralAngle(self, normal1: torch.tensor, normal2: torch.tensor):
        return torch.abs(
            torch.dot(normal1, normal2) /
            (torch.norm(normal1) * torch.norm(normal2)))
