from options.test_options import TestOptions
from data import DataLoader
from util.writer import Writer
import torch
import os
from model import network


if __name__ == '__main__':
    opt = TestOptions().parse()
    dataset = DataLoader(opt)  # input data
    dataset_size = len(dataset)
    print("testing data size is %d" % dataset_size)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    mynet = torch.load(os.path.join(save_dir, "latest_net.pkl"))
    print(mynet)
    lossSD = network.LossSymmetryDistance()
    lossR = network.LossRegularization()
    validate = network.validateOutputs()

    # test begins
    for i, sample in enumerate(dataset, 0):
        voxel = sample['voxel']
        print('voxel', voxel.shape)
        outputs = mynet(voxel)
        lsd = lossSD(outputs, sample)
        lr = lossR(outputs)
        outputs = outputs.view(6, 4)
        lsd = lsd.view(6)
        lr = lr.view(1)
        outputs = validate(outputs, lsd, opt.maxlsd, opt.maxcosval)
        log = "{}th sample, lsd4plane is {} and lr4axis is {}, validated outputs is {}\n".format(i, lsd, lr, outputs)
        print(log)  # when test use > your opt path.txt 2 save your log data
        '''if not opt.accGTE:
            #save lsd to a file.
        else:
            accGTE = acc(outputs)
            #save accGTE to a file too.

def acc():
  # calculate the GTE error
  #GTE = (ai−agt)2 + (bi−bgt)2 + (ci−cgt)2 + (di−dgt)2
'''

