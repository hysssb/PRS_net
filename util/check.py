import os

dataroot = 'D:\\python\\PRS_NET_master\\datasets\\check\\'
for index in range(1,150):
    name = str(index) + '.pcd'
    NewPcdPath = os.path.join(dataroot, name)
    num = 0
    with open(NewPcdPath, mode='r') as pcdFile:
        for i in range(10):
            lines = pcdFile.readline()
        for i in range(1000):
            line = pcdFile.readline()
            if line != '':
                num += 1
            else:
                break
    if num != 1000:
        print(index, 'wrong')
    else:
        print(index, '!!!!!')
