

import argparse

import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataloader3 import ViPCDataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import params
from models.SVDDual54 import Model
from decoder.utils.utils import *
from datetime import datetime, timedelta
import time
import logging
from models.model_utils import PCViews
from visualization import *

parser = argparse.ArgumentParser()

parser.add_argument('--test_category', type=str, default='plane')
parser.add_argument('--Test_record_path', type=str, default='/data/home-gxu/ly21/XMFnet/log/test_vis/plane/Test_record')

args = parser.parse_args()

# ���в��Գ���ʱ�ǵ��޸���������
# test_category = 'watercraft'
# Test_record_path = '/home-gxu/ly21/XMFnet_A100/log/models_svd_dual6/models_svd_dual6_0.1_96_watercraft_train36_Wed Nov  8 12:13:56 2023/Test_record'


# ��ӡ���ն˲���¼����־
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
ViPCDataset_test = ViPCDataLoader('test_list2.txt', data_path=opt.dataroot, status="test", category=args.test_category)
test_loader = DataLoader(ViPCDataset_test,
                         batch_size=80,
                         num_workers=2,
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
    for data in tqdm(test_loader, colour='red', desc='Test Process'):
        i += 1

        image = data[0].to(device)
        partial = data[2].to(device)
        gt = data[1].to(device)
        image_path = data[3]

        partial = farthest_point_sample(partial, 2048)
        gt = farthest_point_sample(gt, 2048)

        partial_depth = torch.unsqueeze(render.get_img(partial), 1)

        coarse, complete = model(partial, partial_depth, image)

        # Compute the eval loss
        loss = loss_eval(complete, gt)
        f1, _, _ = loss_f1(complete, gt)
        f1 = f1.mean()

        Loss += loss
        f1_final += f1

        # Visualization
        vis_path = os.path.join(args.Test_record_path, '���ӻ�')
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)
            print('�ɹ��������ӻ��ļ���')

        category_path = os.path.join(vis_path, f'{args.test_category.capitalize()}')
        if not os.path.exists(category_path):
            os.mkdir(category_path)
            print(f'�ɹ�����{args.test_category}����ļ��ￄ1�7')

        vis_image_path = os.path.join(category_path, '����ͼ��')
        if not os.path.exists(vis_image_path):
            os.mkdir(vis_image_path)
            print('�ɹ���������ͼ���ļ���')

        vis_input_path = os.path.join(category_path, '������ￄ1�7')
        if not os.path.exists(vis_input_path):
            os.mkdir(vis_dinput_path)
            print('�ɹ�������������ļ��ￄ1�7')

        vis_output_path = os.path.join(category_path, '������ￄ1�7')
        if not os.path.exists(vis_output_path):
            os.mkdir(vis_oeutput_path)
            print('�ɹ�������������ļ��ￄ1�7')

        vis_gt_path = os.path.join(category_path, '��ֵ����')
        if not os.path.exists(vis_gt_path):
            os.mkdir(vis_gt_path)
            print('�ɹ�������ֵ�����ļ���')

        # �洢����ͼ��
        # save_input_image = visualize_image(image_path, vis_image_path, index)

        # �洢���롢�������ֵ����ΪpngͼƬ
        # save_partial_points = visualize_point_cloud(partial, vis_input_path, index)
        # save_complete_points = visualize_point_cloud(complete, vis_output_path, index)
        # save_gt_points = visualize_point_cloud(gt, vis_gt_path, index)
        # index = index + test_loader.batch_size

        # �洢���롢�������ֵ����Ϊobj�ļ�
        # save_partial_points = save_point_cloud_as_ply(partial, vis_input_path, index)
        # save_complete_points = save_point_cloud_as_ply(complete, vis_output_path, index)
        # save_gt_points = save_point_cloud_as_ply(gt, vis_gt_path, index)
        # index = index + test_loader.batch_size

    Loss = Loss/i
    f1_final = f1_final/i

    print_and_log(f'======>The Evaluation Loss for {args.test_category} is :  {Loss:.4f}')
    print_and_log(f'======>The F1-score for {args.test_category} is :  {f1_final:.4f}')
