import torch.utils.data


def CreateDataset(opt):
    """loads datasets class"""

    if opt.dataset_mode == 'ShapeNet':
        from data.ShapeNet_data import ShapeNetData
        dataset = ShapeNetData(opt)
    elif opt.dataset_mode == 'testSet':
        from data.testSet_data import testSetData
        dataset = testSetData(opt)
    return dataset


class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads)
        )  # shuffle false:order true: randomly

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
