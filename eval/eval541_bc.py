# encoding=gbk
import argparse

import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataloader5 import ViPCDataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import params
from models.SVDDual54_bc import Model
from decoder.utils.utils import *
from datetime import datetime, timedelta
import time
import logging
from models.model_utils import PCViews
from visualization import *

parser = argparse.ArgumentParser()

parser.add_argument('--test_category', type=str, default='plane')
parser.add_argument('--Test_record_path', type=str, default='/data/home-gxu/ly21/XMFnet/log/models_svd_dual54_bc/models_svd_dual54_bc_0.1_160_plane_train54_bc_Mon Jan  8 10:14:14 2024/Test_record')

args = parser.parse_args()

# 运行测试程序时记得修改下列配置
# test_category = 'watercraft'
# Test_record_path = '/home-gxu/ly21/XMFnet_A100/log/models_svd_dual6/models_svd_dual6_0.1_96_watercraft_train36_Wed Nov  8 12:13:56 2023/Test_record'


# 打印到终端并记录到日志
def print_and_log(message):
    logging.info(message)
    print(message)


log_format = "%(asctime)s - %(levelname)s: %(message)s"
if not os.path.exists(args.Test_record_path):
    os.makedirs(args.Test_record_path)
logging.basicConfig(filename=os.path.join(args.Test_record_path, f'test_{args.test_category}.log'), filemode='a+', level=logging.INFO, format=log_format)
# checkpoint = os.path.join(args.Test_record_path, 'best_model_watercraft.pth')
checkpoint = os.path.join(args.Test_record_path, f'best_model_{args.test_category}.pth')

opt = params()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
ViPCDataset_test = ViPCDataLoader('24view.txt', data_path=opt.dataroot, status="test", category=args.test_category)
test_loader = DataLoader(ViPCDataset_test,
                         batch_size=1,
                         num_workers=1,
                         shuffle=False,
                         drop_last=True)
#####
model = Model().to(device)
#####

print_and_log('Test Start')
print_and_log(f'Batch size: {test_loader.batch_size}')
print_and_log("load checkpoint ===>" + checkpoint)
model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

render = PCViews(TRANS=-1.5, RESOLUTION=224)

loss_eval = L2_ChamferEval_1000()
loss_f1 = F1Score()

with torch.no_grad():
    model.eval()
    i = 0
    index = 0
    Loss = 0
    f1_final = 0
    std_dev_list = []

    for data in tqdm(test_loader, colour='red', desc='Test Process'):
        i += 1
        j = 0
        loss_list = []
        for view in data:
            j += 1
            views, pc, pc_part, image_path = view
            image = views.to(device)
            partial = pc_part.to(device)
            gt = pc.to(device)
            view_path = image_path

            partial = farthest_point_sample(partial, 2048)
            gt = farthest_point_sample(gt, 2048)

            # partial_depth = torch.unsqueeze(render.get_img(partial), 1)

            coarse, complete = model(partial, image)

            # Compute the eval loss
            loss = loss_eval(complete, gt)
            print_and_log(loss)
            print_and_log(view_path)
            f1, _, _ = loss_f1(complete, gt)
            f1 = f1.mean()
            loss_list.append(loss.item())

            # Visualization
            vis_path = os.path.join(args.Test_record_path, '可视化')
            if not os.path.exists(vis_path):
                os.mkdir(vis_path)
                print('成功创建可视化文件夹')

            category_path = os.path.join(vis_path, f'{args.test_category.capitalize()}')
            if not os.path.exists(category_path):
                os.mkdir(category_path)
                print(f'成功创建{args.test_category}类别文件夹')

            vis_image_path = os.path.join(category_path, '输入图像')
            if not os.path.exists(vis_image_path):
                os.mkdir(vis_image_path)
                print('成功创建输入图像文件夹')

            vis_input_path = os.path.join(category_path, '输入点云')
            if not os.path.exists(vis_input_path):
                os.mkdir(vis_input_path)
                print('成功创建输入点云文件夹')

            vis_output_path = os.path.join(category_path, '输出点云')
            if not os.path.exists(vis_output_path):
                os.mkdir(vis_output_path)
                print('成功创建输出点云文件夹')

            vis_gt_path = os.path.join(category_path, '真值点云')
            if not os.path.exists(vis_gt_path):
                os.mkdir(vis_gt_path)
                print('成功创建真值点云文件夹')

            # 存储输入图像
            # save_input_image = visualize_image(view_path, vis_image_path, j)
            #
            # # 存储输入、输出和真值点云
            # save_partial_points = visualize_point_cloud(partial, vis_input_path, j)  # b 2048 3
            # save_complete_points = visualize_point_cloud(complete, vis_output_path, j)  # b 2048 3
            # save_gt_points = visualize_point_cloud(gt, vis_gt_path, j)  # b 2048 3
            # index = index + test_loader.batch_size

        std_dev = np.std(loss_list)
        print_and_log(i)
        print_and_log(f"STD: {std_dev}")
        std_dev_list.append(std_dev)

    max_value = max(std_dev_list)
    max_index = std_dev_list.index(max_value)
    print_and_log(f'max_value: {max_value}')
    print_and_log(f'max_index: {max_index + 1}')
    average_std_dev = np.mean(std_dev_list)
    print_and_log(f"Average STD: {average_std_dev}")
    print_and_log(args.test_category)