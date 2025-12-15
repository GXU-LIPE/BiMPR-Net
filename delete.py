# encoding=gbk
# 输入文件的文件名和要删除的数字串
input_filename = "output3.txt"
output_filename = "output4.txt"
number_to_delete = "02828884"  # 你想要删除的数字串

# 打开输入文件和输出文件
with open(input_filename, "r") as input_file, open(output_filename, "w") as output_file:
    # 逐行读取输入文件
    for line in input_file:
        # 检查行是否包含要删除的数字串
        if number_to_delete not in line:
            print(line)
            # 如果不包含，将行写入输出文件
            output_file.write(line)

# 输出文件现在包含了没有特定数字串的行
