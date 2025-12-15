from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms
from dataloader import ViPCDataLoader
from config import params
from torch.utils.data import DataLoader

opt = params()

ViPCDataset_train = ViPCDataLoader('train_list2.txt', data_path=opt.dataroot, status="train", category='all')

print(f"num of CPU: {mp.cpu_count()}")
for num_workers in range(2, mp.cpu_count(), 2):
    train_loader = DataLoader(ViPCDataset_train,
                              batch_size=64,
                              num_workers=num_workers,
                              shuffle=True,
                              drop_last=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
