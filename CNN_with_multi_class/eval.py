import torch
from data import get_data_split
from data import CaptchaDataset
from model import CaptchaModel
from metrics import acc, multi_acc
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import json
import os
from train import save_history

def eval(model_dir, data_dir, batch_size=64, log_dir='./history', gpu=True):
    """
    :param model_dir: Any,
    :param data_dir: Any,
    :param batch_size: int = 64,
    :param log_dir: str = './history',
    :param gpu: bool = True
    """
    x_test, y_test = get_data_split(data_dir, modes=['test'])
    model = CaptchaModel()
    gpu_available = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if gpu and gpu_available:
        model_state = torch.load(os.path.join(model_dir, 'model-latest.pkl'))
    else:
        model_state = torch.load(os.path.join(model_dir, 'model-latest.pkl'), map_location=lambda storage, loc: storage)

    model.load_state_dict(model_state['network'])
    test_dataset = CaptchaDataset((x_test, y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    model.eval()

    acc_history = []
    with tqdm(total=int(np.ceil(len(test_loader.dataset) / batch_size)), desc='Eval') as eval_bar:
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred_1, pred_2, pred_3, pred_4 = model(x)
            acc_total = acc(pred_1, y[:, 0]) + acc(pred_2, y[:, 1] + acc(pred_3, y[:, 2]) + acc(pred_4, y[:, 3]))
            acc_mean = acc_total / 4.

            pred = torch.stack((pred_1, pred_2, pred_3, pred_4), dim=-1)
            multi_acc_mean = multi_acc(torch.argmax(pred, dim=1), y)
            acc_history.append([acc_mean, multi_acc_mean])
            
            eval_bar.update()
            eval_bar.set_postfix(acc=acc_mean, multi_acc=multi_acc_mean)
    
    save_history('eval.json', acc_history, log_dir)

import click

@click.command()
@click.help_option('-h', '--help')
@click.option('-o', '--model_dir', default='./model', type=click.Path(), help='The model dir to save models or load models', required=False)
@click.option('-i', '--data_dir', default='./captchas', type=click.Path(), help='the path of dataset', required=False)
@click.option('-b', '--batch_size', default=128, type=int, help='batch size', required=False)
@click.option('-l', '--log_dir', default='./history', type=click.Path(), help='The log_file path', required=False)
@click.option('-u', '--use_gpu', default=True, type=bool, help='calculate by gpu', required=False)
def read_cli(model_dir, data_dir, batch_size, log_dir, use_gpu):
    eval(model_dir=model_dir, data_dir=data_dir, batch_size=batch_size, log_dir=log_dir, gpu=use_gpu)


if __name__ == '__main__':
    read_cli()