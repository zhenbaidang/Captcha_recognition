import os
import numpy as np
from sklearn.model_selection import train_test_split


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
    image_names = os.listdir(path)
    image_names = [name for name in image_names if name.endswith(".jpg")]
    label2id, id2label = get_dict()
    results = [[label2id[name[:1]], label2id[name[1:2]], label2id[name[2:3]], label2id[name[3:4]]] for name in image_names]
    image_names = [os.path.join(path, name) for name in image_names]

    return image_names, np.