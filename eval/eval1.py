# encoding=gbk

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataloader import ViPCDataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import params
from model1 import Network1
from decoder.utils.utils import *
from datetime import datetime, timedelta
import time
import logging
from visualization import *


##### 运行测试程序时记得修改下列配置
test_category = 'plane'
Test_record_path = '/data/home-gxu/ly21/XMFnet/log/model1_Network1/model1_Network1_1.0_128_plane_train1_Sat Apr  1 13:47:26 2023/Test_record'


# 打印到终端并记录到日志
def print_and_log(message):
    logging.info(message)
    print(message)


log_format = "%(asctime)s - %(levelname)s: %(message)s"
if not os.path.exists(Test_record_path):
    os.makedirs(Test_record_path)
logging.basicConfig(filename=os.path.join(Test_record_path, f'test_{test_category}.log'), filemode='w+', level=logging.INFO, format=log_format)
checkpoint = os.path.join(Test_record_path, f'best_model_{test_category}.pth')

opt = params()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
ViPCDataset_test = ViPCDataLoader('test_list2.txt', data_path=opt.dataroot, status="test", category=test_category)
test_loader = DataLoader(ViPCDataset_test,
                         batch_size=16,
                         num_workers=0,
                         shuffle=True,
                         drop_last=True)
#####
model = Network1().to(device)
#####

print_and_log('Test Start')
print_and_log(f'Batch size: {test_loader.batch_size}')
print_and_log("load checkpoint ===>" + checkpoint)
model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

loss_eval = L2_ChamferEval_1000()
loss_f1 = F1Score()

with torch.no_grad():
    model.eval()
    i = 0
    index = 0
    Loss = 0
    f1_final = 0
    for data in tqdm(test_loader, colour='red', desc='Test Process'):
        i += 1

        image = data[0].to(device)
        partial = data[2].to(device)
        gt = data[1].to(device)  

        partial = farthest_point_sample(partial, 2048)
        gt = farthest_point_sample(gt, 2048)
    
        partial = partial.permute(0, 2, 1)

        complete = model(partial, image)

        # Compute the eval loss
        loss = loss_eval(complete, gt)
        f1, _, _ = loss_f1(complete, gt)
        f1 = f1.mean()

        Loss += loss
        f1_final += f1

        # Visualization
        vis_path = os.path.join(Test_record_path, '可视化')
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)
            print('成功创建可视化文件夹')

        vis_image_path = os.path.join(vis_path, '输入图像')
        if not os.path.exists(vis_image_path):
            os.mkdir(vis_image_path)
            print('成功创建输入图像文件夹')

        vis_input_path = os.path.join(vis_path, '输入点云')
        if not os.path.exists(vis_input_path):
            os.mkdir(vis_input_path)
            print('成功创建输入点云文件夹')

        vis_output_path = os.path.join(vis_path, '输出点云')
        if not os.path.exists(vis_output_path):
            os.mkdir(vis_output_path)
            print('成功创建输出点云文件夹')

        vis_gt_path = os.path.join(vis_path, '真值点云')
        if not os.path.exists(vis_gt_path):
            os.mkdir(vis_gt_path)
            print('成功创建真值点云文件夹')

        # 存储输入图像
        save_input_image = visualize_image(image, vis_image_path, index)

        # 存储输入、输出和真值点云
        save_partial_points = visualize_point_cloud(partial.permute(0, 2, 1), vis_input_path, index)
        save_complete_points = visualize_point_cloud(complete, vis_output_path, index)
        save_gt_points = visualize_point_cloud(gt, vis_gt_path, index)
        index = index + test_loader.batch_size

    Loss = Loss/i
    f1_final = f1_final/i

    print_and_log(f'======>The Evaluation Loss for {test_category} is :  {Loss:.4f}')
    print_and_log(f'======>The F1-score for {test_category} is :  {f1_final:.4f}')
