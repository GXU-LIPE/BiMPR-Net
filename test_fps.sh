#!/bin/bash

# 定义参数数组
partial_points=(2048 1024 256)

# 循环遍历参数数组
for partial_point in "${partial_points[@]}"; do
    # 打印当前参数值
    echo "Running script with partial_point = $partial_point"

    # 调用 Python 脚本并传入参数
    python eval54_fps.py --partial_point $partial_point

    # 如果你的脚本需要等待一段时间再运行，可以在此添加 sleep 命令
    sleep 20
done
