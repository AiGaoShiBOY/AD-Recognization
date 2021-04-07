# 这是一个读取数据的函数
import csv
from datetime import datetime
import os
import re

file_name = 'ADNI1_Baseline_3T_4_06_2021.csv'

data_list = []
label_list = []

with open(file_name) as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        # 获取病人的姓名
        name = row[1]
        # 处理MRI的方法
        method = row[7].replace(';', '_')
        method = method.replace(' ', '_')
        label = row[2]
        d = datetime.strptime(row[9], "%m/%d/%Y")
        d = datetime.strftime(d, '%Y-%m-%d')
        # 进入子目录
        # 首先获取根目录
        root = os.getcwd()
        route = "ADNI\\"+ name + '\\' + method
        route1 = os.path.join(root, route)
        # 对于每个病人，遍历他的对应方法的文件夹
        for dirname in os.listdir(route1):
            # 寻找合适的日期
            match_obj = re.match(d, dirname)
            if match_obj:
                route2 = os.path.join(route1, dirname)
                for root, dirs, files in os.walk(route2):
                    cur_root = root
                # 得到了.nii文件的地址
                path = os.path.join(cur_root, files[0])
                data_list.append(path)
                if label == 'CN':
                    label_list.append(0)
                elif label == 'AD':
                    label_list.append(1)
                elif label == 'MCI':
                    label_list.append(2)
                break


def get_data_list():
    return data_list, label_list










