from tqdm import tqdm
import time

data = range(10)

# 使用 tqdm 迭代对象并显示进度条
batch_bar = tqdm(data)
i = 200
for item in batch_bar:
    # 模拟任务的耗时操作
    time.sleep(0.5)
    i += 1
    # 设置进度条的附加信息
    batch_bar.set_postfix(loss=i, accuracy=0.9)
