import torch
import torchvision.models as models
import torch.nn as nn



class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        # 加载预训练的 VGG16 模型
        vgg16 = models.vgg16(pretrained=True)

        # 冻结 features 层的参数
        for param in vgg16.features.parameters():
            param.requires_grad = False

        vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-1])
        self.vgg16 = vgg16
        self.out1 = nn.Linear(4096, 62)
        self.out2 = nn.Linear(4096, 62)
        self.out3 = nn.Linear(4096, 62)
        self.out4 = nn.Linear(4096, 62)
    
    def forward(self, x):
        x = self.vgg16(x)
        y_1 = self.out1(x)
        y_2 = self.out2(x)
        y_3 = self.out3(x)
        y_4 = self.out4(x)
        return y_1, y_2, y_3, y_4
