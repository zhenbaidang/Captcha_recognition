# ZhenBot，为RPA系统开发的验证码自动识别机器人

为降低部署落地成本，接口部分以最小改动原则，在ddddocr项目基础上进行修改。感谢原作者。

**ZhenBot ** 项目架构：

```.
├── Dockerfile # 用于构建Docker镜像，服务启动命令部分添加参数“--slide”以启动滑块检测模型
├── README.md 
├── api_server.py # 对应原项目ocr_api_server脚本
├── requirements.txt # 项目依赖
└── zhenbot # zhenbot核心代码库
    ├── __init__.py 
    ├── best.onnx # 滑块检测模型参数预训练文件
    ├── common.onnx # 原项目识别模型推理参数
    ├── common_det.onnx # 原项目检测模型推理参数
    └── common_old.onnx # 原项目识别模型推理参数（旧）
```

注意：由于依赖中包含opencv，本项目首次Docker镜像创建时间可能较长。

## Version 1.0

已实现：

1. 保留ddddocr的ocr及det部分（Dockerfile中对应参数为`--ocr` 及 `--det`）(未经测试，建议保留原识别服务)

2. zhenbot核心代码部分：

    - 为`ZhenBot`类添加`slide_inference(self, im, im0)`实例方法
        接受输入参数：im: tensor；im0: 经opencv解码的原始图像array数据
        返回值为5个浮点数字构成的字符串，分别对应

        > （1）滑块检测框bbox中心坐标X / 原始图像宽度Width
        >
        > （2）滑块检测框bbox中心坐标Y / 原始图像高度Height
        >
        > （3）滑块检测框bbox的宽度width / 原始图像宽度Width
        >
        > （4）滑块检测框bbox的高度height / 原始图像高度Height
        >
        > （5）模型预测该bbox的置信度

    - 添加全局方法`img_bs64_to_det_model_input_tensor(input_data, input_type='base64', fp16=False)`
        接受输入参数：`input_data`, ` input_type='base64'`

        > input_type: ['base64', 'img_bytes']
        >
        > Input_data: 当input_type为base64时，方法接受base64字符串作为输入；当input_type为img_bytes时，方法接受已经过b64decode的图像字节数据img_bytes作为输入

        返回值为：im, im0

        > im: 经预处理的图像Tensor，形状为[batch_size, channel, height, width]。对于单张彩色图片，该形状固定为[1, 3, 按原图比例缩放的高度，640] （预处理阶段将图像长边调整为640，短边保持比例缩放）
        >
        > im0: 经base64解码及opencv通道转换为RGB后的原始图像，im0的shape属性返回tuple: (Height, Width, channel)

    - 添加`non_max_suppression()`, `letterbox()` 等其他全局工具方法

3. api_server部分 

    - 服务初始化及启动部分：

        > 添加参数`--slide`以启动滑块检测模块，其余保持不变，端口号默认保持9898
        >
        > 添加全局方法`get_img_for_slide(request, img_type='b64', img_name='image')`
        > 该方法目前仅接受base64传入方式，经解码并调用zhenbot全局方法`img_bs64_to_det_model_input_tensor`得到im, im0作为返回值，作为`Server.slide_inference(im, im0)`方法的输入

    - 路由配置及服务调用部分：`@app.route('/<opt>/<img_type>/<ret_type>', methods=['POST'])`

        >弃用原Server类中的`slide()`方法，将滑块检测模块整合到ocr()方法中，具体调用类型依据路由参数中的<opt>进行区分。opt可选参数为[ocr, det, slide]，其中ocr、det保持不变（未经测试），**当opt选择slide时，img_type必须使用b64**
        >
        >ret_type留空默认值为text
        >返回字符串中，在slide_inference()方法返回值的基础上，追加了原始图像宽度Width及原始图像高度Height两个值，共7个value

Todo:

重构识别部分代码

### Example

Dockerfile中的启动参数：
`CMD ["python3", "api_server.py", "--port", "9898", "--ocr", "--det", "--slide"]`

端口号此处可进行修改，默认保持9898
可仅使用--slide参数启动滑块检测服务

调用实例：

```python
import requests
import base64
import json

# # Read image file and encode to base64
# with open("image.jpg", "rb") as image_file:
#     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
import time
start = time.time()
img_base64 = 'iVBORw0KGgoAAAANSUhE...UVORK5CYII=' # for example
# Prepare data for POST request
data = json.dumps({ "image": img_base64 })

# Send POST request to server
response = requests.post('http://127.0.0.1:9898/slide/b64', data=data, headers={'Content-Type': 'application/json'})
print(f'time: {time.time() - start:.4f}s')
# Print the response
print(response.text)

```

输出：

> time: 0.5944s 
> 0.531944 0.746429 0.136111 0.35 0.9756 140 360