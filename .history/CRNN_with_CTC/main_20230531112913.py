from data import get_data_split
x_test, y_test = get_data_split('/Users/liyaoting/Desktop/毕业设计论文/Captcha_Dataset/captchas', save=False, modes=['test'])
print(y_test)