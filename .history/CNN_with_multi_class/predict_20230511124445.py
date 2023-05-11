import torch
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor

from data import get_dict
from model import CaptchaModel
def predict(captcha, model_dir='./model', gpu=True):
    """

    :param captcha:
    :param model_dir:
    :param gpu:
    :param mode:
    :return:
    """
    gpu_available = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CaptchaModel()
    model = model.to(device)

    if gpu and gpu_available:
        model_state = torch.load(os.path.join(model_dir, 'model-latest.pkl'))
    else:
        model_state = torch.load(os.path.join(model_dir, 'model-latest.pkl'), map_location=lambda storage, loc: storage)

    model.load_state_dict(model_state['network'])


    transformer = Compose(ToTensor())

    img_pil = Image.open(captcha)
    img_tensor = transformer.transforms(img_pil)

    model.eval()
    x = torch.stack([img_tensor])
    x = x.to(device)
    pred1, pred2, pred3, pred4 = model(x)

    pred_seq = [torch.argmax(pred1).item(), torch.argmax(pred2).item(), torch.argmax(pred3).item(), torch.argmax(pred4).item()]
    pred_seq = [item + 1 for item in pred_seq]

    _, id2label = get_dict()

    res = ''.join([id2label[i] for i in pred_seq])

    return res

import click

@click.command()
@click.help_option('-h', '--help')
@click.option('-i', '--image_path', type=click.Path(), help='The path of the captcha image', required=True)
@click.option('-o', '--model_dir', default='./model', type=click.Path(), help='The model dir to save models or load models', required=False)
@click.option('-u', '--use_gpu', default=False, type=bool, help='Train by gpu or cpu', required=False)
def read_cli(image_path, model_dir, mode, use_gpu):
    """

    :param image_path:
    :param model_dir:
    :param use_gpu:
    :return:
    """
    res = predict(image_path, model_dir, use_gpu)
    print('The result of the captcha is: ' + str(res))

if __name__ == "__main__":
    read_cli()