import os
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CRNN
from data import get_data_split, CaptchaDataset
from metrics import transposition
def save_history(filename, history, history_path):
    if not os.path.exists(history_path):
        os.mkdir(history_path)
    out_file = os.path.join(history_path, filename)
    with open(out_file, 'w', encoding='utf-8') as out_fp:
        json.dump(history, out_fp)


def load_history(filename, history_path):
    filepath = os.path.join(history_path, filename)
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as file:
        history = json.load(file)
    return history

def train(path, split=[6, 1, 1], batch_size=64, epochs=100, learning_rate=0.001, initial_epoch=0, save_frequency=2, model_dir='./model', log_dir='./history', continue_pkl=None, gpu=True):
    """
    :param path: Any,
    :param split: Any = [6, 1, 1],
    :param batch_size: int = 64,
    :param epochs: int = 100,
    :param learning_rate: float = 0.001,
    :param initial_epoch: int = 0,
    :param save_frequency: int = 2,
    :param model_dir: str = './model',
    :param log_dir: str = './history',
    :param continue_pkl: Any | None = None,
    :param gpu: bool = True
    """

    if not os.path.exists(path):
        raise FileNotFoundError('训练数据未找到')

    x_train, y_train, x_dev, y_dev = get_data_split(
        path, split=split, modes=['train', 'dev'])
    train_dataset = CaptchaDataset((x_train, y_train), shuffle=True)
    dev_dataset = CaptchaDataset((x_dev, y_dev))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

    gpu_available = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(imgH=32, num_channel=3, nclass=62+1, hidden_size=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss(blank=0)

    model = model.to(device)
    

    # 从已保存的状态中加载模型
    if continue_pkl is not None and os.path.exists(os.path.join(model_dir, continue_pkl)):
        if gpu and gpu_available:
            initial_state = torch.load(os.path.join(model_dir, continue_pkl))
        else:
            initial_state = torch.load(os.path.join(
                model_dir, continue_pkl), map_location=lambda storage, loc: storage)
        model.load_state_dict(initial_state['network'])
        optimizer.load_state_dict(initial_state['optimizer'])
        initial_epoch = initial_state['epoch'] + 1

    elif continue_pkl is not None and os.path.exists(os.path.join(model_dir, 'model-latest.pkl')):
        if gpu and gpu_available:
            latest_state = torch.load(
                os.path.join(model_dir, 'model-latest.pkl'))
        else:
            latest_state = torch.load(os.path.join(
                model_dir, 'model-latest.pkl'), map_location=lambda storage, loc: storage)
        model.load_state_dict(latest_state['network'])
        optimizer.load_state_dict(latest_state['optimizer'])
        initial_epoch = latest_state['epoch'] + 1

    elif continue_pkl is not None and initial_epoch is not None and os.path.exists(os.path.join(model_dir, 'model-%d.pkl' % initial_epoch)):
        if gpu and gpu_available:
            initial_state = torch.load(os.path.join(
                model_dir, 'model-%d.pkl' % initial_epoch))
        else:
            latest_state = torch.load(os.path.join(
                model_dir, 'model-%d.pkl' % initial_epoch), map_location=lambda storage, loc: storage)
        model.load_state_dict(initial_state['network'])
        optimizer.load_state_dict(initial_state['optimizer'])
        initial_epoch = initial_state['epoch'] + 1
    else:
        initial_epoch = 0

    batch_history_train = load_history(
        filename='history_batch_train.json', history_path=log_dir)
    epoch_history_train = load_history(
        filename='history_epoch_train.json', history_path=log_dir)
    epoch_history_dev = load_history(
        filename='history_epoch_dev.json', history_path=log_dir)
    batch_history_train = batch_history_train[:initial_epoch]
    epoch_history_train = epoch_history_train[:initial_epoch]
    epoch_history_dev = epoch_history_dev[:initial_epoch]

    with tqdm(total=epochs, desc='Epoch', initial=initial_epoch) as epoch_bar:
        for epoch in range(initial_epoch, epochs):
            model.train()
            loss_batchs = []
            acc_batchs = []
            multi_acc_batchs = []
            # train 一个epoch
            with tqdm(total=int(np.ceil(len(train_loader.dataset) / batch_size)), desc='Batch') as batch_bar:
                for batch, (x, y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    x = x.to(device)
                    y = y.to(device)
                    
                    output = model(x) # output.shape = (weight(seq_len), batch_size, num_class)
                    output_lengths = torch.full((output.shape[1],), output.shape[0], dtype=torch.long).to(device)
                    y_lengths = torch.full((y.shape[0],), y.shape[1], dtype=torch.long).to(device) # the length of label sequence
                    loss = criterion(output.log_softmax(2), y, output_lengths, y_lengths)

                    predict = transposition(output)
                    

                    # acc_total:四个字符位置的正确率之和
                    # acc按字符正确率进行统计，把batch*class_num形状的pred在最后一维上计算argmax，和该位置字符的y相比，求平均正确率
                    acc_total = acc(
                        pred_1, y[:, 0]) + acc(pred_2, y[:, 1]) + acc(pred_3, y[:, 2]) + acc(pred_4, y[:, 3])
                    # acc_mean:统计平均字符正确率
                    acc_mean = acc_total / 4.
                    # stack会在指定维度添加一维（ndim+1）
                    pred = torch.stack(
                        (pred_1, pred_2, pred_3, pred_4), dim=-1)
                    multi_acc_mean = multi_acc(torch.argmax(pred, dim=1), y)

                    loss_batchs.append(loss_total.item())
                    acc_batchs.append(acc_mean)
                    multi_acc_batchs.append(multi_acc_mean)

                    batch_bar.set_postfix(loss=loss_total.item(
                    ), acc=acc_mean, multi_acc=multi_acc_mean)
                    batch_bar.update()
                    batch_history_train.append(
                        [loss_total.item(), acc_mean, multi_acc_mean])
                    save_history('history_batch_train.json',
                                 batch_history_train, log_dir)

                    loss_total.backward()
                    optimizer.step()
            epoch_bar.set_postfix(loss_mean=np.mean(loss_batchs), acc_mean=np.mean(
                acc_batchs), multi_acc_mean=np.mean(multi_acc_batchs))
            epoch_bar.update()
            epoch_history_train.append([np.mean(loss_batchs).item(), np.mean(
                acc_batchs).item(), np.mean(multi_acc_batchs).item()])
            save_history('history_epoch_train.json',
                         epoch_history_train, log_dir)

            # validate 验证一个epoch
            with tqdm(total=int(np.ceil(len(dev_loader.dataset) / batch_size)), desc='Val Batch') as batch_bar:
                model.eval()
                loss_batchs_dev = []
                acc_batchs_dev = []
                multi_acc_batchs_dev = []
                for batch, (x, y) in enumerate(dev_loader):
                    x = x.to(device)
                    y = y.to(device)
                    pred_1, pred_2, pred_3, pred_4 = model(x)
                    loss_1, loss_2, loss_3, loss_4 = criterion(pred_1, y[:, 0]), criterion(
                        pred_2, y[:, 1]), criterion(pred_3, y[:, 2]), criterion(pred_4, y[:, 3])
                    loss_total = loss_1 + loss_2 + loss_3 + loss_4
                    acc_total = acc(
                        pred_1, y[:, 0]) + acc(pred_2, y[:, 1]) + acc(pred_3, y[:, 2]) + acc(pred_4, y[:, 3])
                    # acc_mean:统计平均字符正确率
                    acc_mean = acc_total / 4.
                    # stack会在指定维度添加一维（ndim+1）
                    pred = torch.stack(
                        (pred_1, pred_2, pred_3, pred_4), dim=-1)
                    multi_acc_mean = multi_acc(torch.argmax(pred, dim=1), y)

                    loss_batchs_dev.append(loss_total.item())
                    acc_batchs_dev.append(acc_mean)
                    multi_acc_batchs_dev.append(multi_acc_mean)

                    batch_bar.set_postfix(loss=loss_total.item(
                    ), acc=acc_mean, multi_acc=multi_acc_mean)
                    batch_bar.update()
                epoch_history_dev.append([np.mean(loss_batchs_dev).item(), np.mean(
                    acc_batchs_dev).item(), np.mean(multi_acc_batchs_dev).item()])
                save_history('history_epoch_dev.json',
                             epoch_history_dev, log_dir)

            # 保存模型
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            state_dict = {
                'network': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            if epoch % save_frequency == 0:
                model_path = os.path.join(model_dir, 'model-%d.pkl' % epoch)
                torch.save(state_dict, model_path)

            torch.save(state_dict, os.path.join(model_dir, 'model-latest.pkl'))


# Create an instance of the CRNN model
crnn = CRNN(imgH=32, nc=3, nclass=37, nh=256)

# Move your model to the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
crnn = crnn.to(device)

# Initialize the CTC Loss function
criterion = nn.CTCLoss(blank=0)

# Let's say you have a batch of 20 images, each of size (3, 32, 128)
inputs = torch.randn(20, 3, 32, 128).to(device)

# The length of the output sequence produced by the CRNN
input_lengths = torch.full((20,), 31, dtype=torch.long).to(device) # the length of output sequence, for example here is 31

# Target sequences, in this case let's say we have sequences of length 5
targets = torch.randint(1, 37, (20, 5), dtype=torch.long).to(device) # assume the label sequence length is 5

# The length of the target sequences
target_lengths = torch.full((20,), 5, dtype=torch.long).to(device) # the length of label sequence, for example here is 5

# Forward pass through the CRNN
outputs = crnn(inputs)

# Compute the CTC loss
loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)

# Backward pass and optimization
# (assuming you have an optimizer defined as `optimizer`)
loss.backward()
optimizer.step()
