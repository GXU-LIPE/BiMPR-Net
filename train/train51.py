# encoding=gbk
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
from decoder.utils.utils import *
from models.SVDDual51 import Model
from config import params
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import ViPCDataLoader
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import logging
from models.model_utils import PCViews
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp

opt = params()

MODEL = 'models_svd_dual51'
FLAG = 'train51'
DEVICE = 'cuda:0'
VERSION = '0.1'
CLASS = 'cabinet'
BATCH_SIZE = 16
MAX_EPOCH = 400
EVAL_EPOCH = 1
RESUME = False

TIME_FLAG = time.asctime(time.localtime(time.time()))  # 获取当前时间并将其转换为可读的字符串格式
CKPT_RECORD_FOLDER = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/Test_record'  # 日志文件的保存路径
TRAIN_RECORD = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}'  # 训练日志路径
CKPT_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt.pth'  # 模型的保存路径
CONFIG_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/config.txt'  # 配置文件的保存路径

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用三元表达式判断CUDA是否可用，如果可用则将设备设置为cuda:0，否则设置为cpu
device_ids = None

if torch.cuda.is_available():
    device_ids = list(range(torch.cuda.device_count()))

if not os.path.exists(os.path.join(TRAIN_RECORD)):
    os.makedirs(os.path.join(TRAIN_RECORD))
logging.basicConfig(filename=os.path.join(TRAIN_RECORD, f'train_{CLASS}.log'), filemode='w+', level=logging.INFO,
                    format="%(asctime)s - %(levelname)s: %(message)s")


def save_record(epoch, prec1, net: nn.Module):  # 将当前的神经网络状态字典保存到文件中
    state_dict = net.state_dict()
    torch.save(state_dict, os.path.join(CKPT_RECORD_FOLDER, f'epoch{epoch}_{prec1:.4f}.pth'))


# 打印到终端并记录到日志
def print_and_log(message):
    logging.info(message)
    print(message)


def save_ckpt(epoch, net, optimizer_all):  # 将当前的神经网络状态字典和优化器状态字典保存到文件中
    ckpt = dict(
        epoch=epoch,
        model=model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        optimizer_all=optimizer.state_dict(),
    )
    torch.save(ckpt, CKPT_FILE)


def set_seed(seed=42):  # 设置随机数种子，以便复现结果
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # 设置当前GPU
            torch.cuda.manual_seed_all(seed)  # 设置所有GPU
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True


def weights_init_normal(m):
    """ Weights initialization with normal distribution.. Xavier """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train_one_step(data, optimizer, network):
    image = data[0].to(device)
    partial = data[2].to(device)
    gt = data[1].to(device)

    partial = farthest_point_sample(partial, 2048)  # B 2048 3
    gt = farthest_point_sample(gt, 2048)

    partial_depth = torch.unsqueeze(render.get_img(partial), 1)

    # partial = partial.permute(0, 2, 1)
    network.train()
    coarse, complete = network(partial, partial_depth, image)

    loss_total = loss_cd(complete, gt)

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    return loss_total


best_loss = 99999
best_epoch = 0
resume_epoch = 0
render = PCViews(TRANS=-1.5, RESOLUTION=224)
board_writer = SummaryWriter(comment=f'{MODEL}_{VERSION}_{BATCH_SIZE}_{FLAG}_{CLASS}_{TIME_FLAG}')

#####
model = Model()
if device_ids is not None and len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

print_and_log('Model Structure:')
print_and_log(model)
#####

# learning_rate = 0.0001
# decay_rate = 0.98
# decay_step = 2

loss_cd = L1_ChamferLoss()
loss_cd_eval = L2_ChamferEval()
optimizer = torch.optim.Adam(filter(
    lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999))
# scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)

ViPCDataset_train = ViPCDataLoader('train_list2.txt', data_path=opt.dataroot, status="train", category=CLASS)
train_loader = DataLoader(ViPCDataset_train,
                          batch_size=BATCH_SIZE,
                          num_workers=16,
                          shuffle=True,
                          drop_last=True)

ViPCDataset_test = ViPCDataLoader('test_list2.txt', data_path=opt.dataroot, status="test", category=CLASS)
test_loader = DataLoader(ViPCDataset_test,
                         batch_size=BATCH_SIZE,
                         num_workers=16,
                         shuffle=True,
                         drop_last=True)

if RESUME:
    ckpt_path = "/data/home-gxu/ly21/XMFnet/log/models_svd_dual6/models_svd_dual6_0.1_192_plane_train36_Mon Oct 23 07:51:19 2023/Test_record/best_model_plane.pth"
    ckpt_dict = torch.load(ckpt_path)
    model.load_state_dict(ckpt_dict['model_state_dict'])
    optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
    resume_epoch = ckpt_dict['epoch']
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

if not os.path.exists(os.path.join(CKPT_RECORD_FOLDER)):
    os.makedirs(os.path.join(CKPT_RECORD_FOLDER))  # 创建文件夹

with open(CONFIG_FILE, 'w') as f:  # 打开配置文件，写入各个参数
    f.write('RESUME:' + str(RESUME) + '\n')
    f.write('FLAG:' + str(FLAG) + '\n')
    f.write('DEVICE:' + str(DEVICE) + '\n')
    f.write('BATCH_SIZE:' + str(BATCH_SIZE) + '\n')
    f.write('MAX_EPOCH:' + str(MAX_EPOCH) + '\n')
    f.write('CLASS:' + str(CLASS) + '\n')
    f.write('VERSION:' + str(VERSION) + '\n')
    f.write(str(opt.__dict__))  # 记录所有参数的详细信息，写入配置文件

model.train()
model.to(device)

print_and_log('--------------------')
print_and_log('Training Starting')
print_and_log(f'Training Class: {CLASS}')
print_and_log('--------------------')

set_seed()

decay_rate = 0.98
decay_epochs = 2

for epoch in range(resume_epoch, resume_epoch + opt.n_epochs + 1):
    # if epoch % decay_epochs == 0:
    #     opt.lr = opt.lr * decay_rate
    # else:
    #     opt.lr = opt.lr
    if epoch < 25:
        opt.lr = 0.001
    elif epoch < 125:
        opt.lr = 0.0001
    elif epoch < 225:
        opt.lr = 0.00001
    elif epoch < 325:
        opt.lr = 0.000001
    else:
        opt.lr = 0.0000001

    Loss = 0
    i = 0

    for data in tqdm(train_loader, colour='green', desc=f'{epoch} Training Process'):
        loss = train_one_step(data, optimizer, network=model)
        i += 1
        if i % opt.loss_print == 0:
            board_writer.add_scalar("Loss_iteration", loss.item(), global_step=i + epoch * len(train_loader))
        Loss += loss

    Loss = Loss / i
    print_and_log(f'epoch {epoch} ==> Train Loss = {Loss}, lr ==> {opt.lr}')
    board_writer.add_scalar("Average_Loss_epochs", Loss.item(), epoch)

    if epoch % EVAL_EPOCH == 0:

        with torch.no_grad():
            model.eval()

            i = 0
            Loss = 0

            for data in tqdm(test_loader, colour='yellow', desc=f'{epoch} Validate Process'):
                i += 1
                image = data[0].to(device)
                partial = data[2].to(device)
                gt = data[1].to(device)

                partial = farthest_point_sample(partial, 2048)
                gt = farthest_point_sample(gt, 2048)

                partial_depth = torch.unsqueeze(render.get_img(partial), 1)

                # partial = partial.permute(0, 2, 1)
                coarse, complete = model(partial, partial_depth, image)

                loss = loss_cd_eval(complete, gt)

                Loss += loss  # 计算总loss

            Loss = Loss / i  # 计算平均loss
            print_and_log(f'epoch {epoch} ==> Validate Loss = {Loss}')
            board_writer.add_scalar("Average_Loss_epochs_test", Loss.item(), epoch)

            if Loss < best_loss:
                best_loss = Loss
                best_epoch = epoch
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss
                }, os.path.join(CKPT_RECORD_FOLDER, f'best_model_{CLASS}.pth'))
            print()
            print('==> Save The Best Model \n')
            print(best_epoch, ' ', best_loss, '\n')

    print('********************<Best_Model>****************************')
    print_and_log(f'Best_epoch {best_epoch} ==> Best_loss = {best_loss}\n')
    print('************************************************************\n')

    if epoch % opt.ckp_epoch == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': Loss
        }, f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt_{epoch}.pt')

print_and_log('Train Finished!!!')
