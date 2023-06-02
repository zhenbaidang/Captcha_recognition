from torch.utils.data import DataLoader
from data import get_data_split, CaptchaDataset
from metrics import transposition, convert2string_list
import matplotlib.pyplot  as plt

from torchvision import transforms
x_test, y_test = get_data_split('/Users/liyaoting/Downloads/captchas', save=False, modes=['test'])
x_train, y_train, x_dev, y_dev = get_data_split('/Users/liyaoting/Downloads/kaptchas', save=False, split=[6, 1, 1], modes=['train', 'dev'])
train_dataset = CaptchaDataset((x_train, y_train), shuffle=True)
dev_dataset = CaptchaDataset((x_dev, y_dev))
train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=4, shuffle=True)

for batch, (x, y) in enumerate(train_loader):
    print(x.shape)
    for i in range(4):
        a = transforms.ToPILImage()(x[i])
        a.save(f'./{i}.jpg')
    print(convert2string_list(y))
    break
