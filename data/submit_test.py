# -*- coding: utf-8 -*-
import os
import zipfile
from src.config import  *

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


def read_txt_ann(ann_path):
    fp = open(ann_path, encoding='utf-8')
    for line in fp:
        line = line.strip()
        # print(ann_path,line)
        if line != '':
            # if len(line.split('\t')) == 1:
            # print('-' * 10, ann_path, line)
            _, temp_data, words = line.split('\t')
            name, start, end = temp_data.split(' ')
            start, end = int(start), int(end)
            assert (end - start) == len(words)
            # if (end - start) != len(words):
            # print('_' * 10, ann_path, line)


if __name__ == '__main__':
    # zip_src = 'submit.zip'
    # dst_dir = 'temp'
    # unzip_file(zip_src, dst_dir)
    for i in range(1000, 1500):
        txt_path = submit_save_path + '/{}.ann'.format(i)
        # if os.path.exists(txt_path):
        read_txt_ann(txt_path)
        assert 0
