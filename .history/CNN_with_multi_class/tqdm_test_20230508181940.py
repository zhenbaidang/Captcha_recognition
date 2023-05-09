from tqdm import tqdm
import time

# 创建一个可迭代对象（例如列表或 range）
data = range(10)

# 使用 tqdm 迭代对象并显示进度条
for item in tqdm(data):
    # 模拟任务的耗时操作
    time.sleep(0.5)
