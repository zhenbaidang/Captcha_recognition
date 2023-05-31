from data import get_data_split
x_test, y_test = get_data_split('/Users/liyaoting/Downloads/captchas', save=False, modes=['test'])
print(y_test)