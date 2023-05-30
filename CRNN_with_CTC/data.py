import os
import numpy as np
from typing import Any
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def get_dict():
    """
    Get dictionary of id2label and label2id, id2label is a dictionary which indicates the label of an id and the label2id is a reversed from `label2id`
    :return: two dictionaries: label->id, id->label
    """
    label2id = {}
    id2label = {}
    # upper case
    for i in range(26):
        label2id[chr(ord('A') + i)] = 1 + i
        id2label[1 + i] = chr(ord('A') + i)
    # lower case
    for i in range(26):
        label2id[chr(ord('a') + i)] = 1 + i + 26
        id2label[1 + i + 26] = chr(ord('a') + i)
    # numbers
    for i in range(10):
        label2id[chr(ord('0') + i)] = 53 + i
        id2label[53 + i] = chr(ord('0') + i)

    return label2id, id2label

def get_data(path):
    # 给数据集的路径path，返回数据集里图片的文件路径list和一个二维np数组，第一维度是样本，第二维度是label的整个序列每个字符对应字典中的index
    # 注意这里的样本label中的字符个数对于一个数据集来说是写死固定的，这样才能作为np数组进行返回
    # Todo: 后期可以做一个变长的，变长就要做padding以及搞一个target_lengths
    image_names = os.listdir(path)
    image_names = [name for name in image_names if name.endswith(".jpg")]
    labels = [full_name.split('.')[0] for full_name in image_names]
    label2id, id2label = get_dict()
    results = [[label2id[char] for char in label] for label in labels]
    image_names = [os.path.join('./data', name) for name in image_names]

    return image_names, np.array(results, dtype=np.int32) # 这里不进行-1操作（与普通版本的CNN相比），使字典的index从1开始记。CTC的blank占用了index0

def get_data_split(path, split=[6, 1, 1], save=True, out_dir='./data', modes=['train', 'dev', 'test']):
    """
    Get data after split.
    :param path: the path to save images
    :param split: the ratio of train set, dev set and test set
    :param out_dir: the output directory to save data files
    :param modes: the modes at different timestamp, support modes like: (train, dev, test), (train, dev) and (test)
    :return: six data with ratio specified by `split`.
    """
    # 该函数运行后，直接把数据集文件夹path下的所有文件和label分成train、dev、test三块，你的modes里填啥，返回啥而已
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    train_path, dev_path, test_path = os.path.join(out_dir, 'train.npy'), os.path.join(out_dir, 'dev.npy'), os.path.join(out_dir, 'test.npy')
    if os.path.exists(train_path) and os.path.exists(dev_path) and os.path.exists(test_path):

        if 'train' in modes:
            x_train, y_train = np.load(train_path, allow_pickle=True), np.load(os.path.join(out_dir, 'train.y.npy'), allow_pickle=True)
        if 'dev' in modes:
            x_dev, y_dev = np.load(dev_path, allow_pickle=True), np.load(os.path.join(out_dir, 'dev.y.npy'))
        if 'test' in modes:
            x_test, y_test = np.load(test_path, allow_pickle=True), np.load(os.path.join(out_dir, 'test.y.npy'))

    else:
        names, labels = get_data(path)

        ratios = np.array(split) / np.sum(split)

        x_train, x_dev_test, y_train, y_dev_test = train_test_split(names, labels, train_size=ratios[0])
        ratios = np.array(split[1:]) / np.sum(split[1:])
        x_dev, x_test, y_dev, y_test = train_test_split(x_dev_test, y_dev_test, train_size=ratios[0])

        if save:
            np.save(train_path, x_train, allow_pickle=True)
            np.save(os.path.join(out_dir, 'train.y.npy'), y_train, allow_pickle=True)
            np.save(dev_path, x_dev, allow_pickle=True)
            np.save(os.path.join(out_dir, 'dev.y.npy'), y_dev, allow_pickle=True)
            np.save(test_path, x_test, allow_pickle=True)
            np.save(os.path.join(out_dir, 'test.y.npy'), y_test, allow_pickle=True)

    if 'train' in modes and 'dev' in modes and 'test' in modes:
        return  x_train, y_train, x_dev, y_dev, x_test, y_test
    elif 'train' in modes and 'dev' in modes:
        return x_train, y_train, x_dev, y_dev
    elif 'test' in modes:
        return x_test, y_test


class CaptchaDataset(Dataset):
    def __init__(self, data, shuffle=True):
        super(CaptchaDataset, self).__init__()
        self.shuffle = shuffle
        self.x_data = data[0]
        self.y_data = data[1]
        # Resize只填一个int值时，即将图像短边设为该值，对图像进行等比例缩放（32是因为模型要求，32过模型后得1）
        self.image_transformer = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    
    def __getitem__(self, index: Any) -> Any:
        actualIndex = index % self.y_data.shape[0]
        img = Image.open(self.x_data[actualIndex])
        img_tensor = self.image_transformer(img)
        return img_tensor, torch.tensor(self.y_data[actualIndex], dtype=torch.long)
    
    def __len__(self):
        return self.y_data.shape[0]