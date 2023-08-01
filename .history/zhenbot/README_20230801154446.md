# 🤖️ZhenBot: A CAPTCHA Autobot for RPA System

Modified from the **ddddocr** and the **yolov5** project. Thanks to the authors.

## Version 1.0

Achieve：

1. Keep the part of ocr and det.
2. Add slide_inference() instance method whose got im: tensor, im0: img_cv for input and return a string contain five values with space for split in ZhenBot class. The contents are, in order: xy coordinates of the center of the bounding box, the width and height of the bounding box, devided by the width and height of the base64-encoded image, respectively. The final value is confidence of model to predict the bbox. **"bbox_x/img_width bbox_y/img_height bbox_width/img_width bbox_height/img_height confidence of the bbox"**

3. Add a global method in zhenbot lib: img_bs64_to_det_model_input_tensor() to convert the parameter "input_data"(base64 when the parameter "input_type" is "base64"; img_bytes when "input_type" is "img_bytes") to the tensor "im" that has preprocess for model input (shape is (batch, channel, img_height, img_width), for single color pic, always be (1, 3, scaled_height for maintain ratio, fix_scale to 640)) and the raw img array that use cv2 to decode and has "RGB" channel sequence(shape is (raw_img_height, raw img_width, channel)).

4. Add argument "--slide" to api_server.py for activate slide-inference model
5. Add instance method "slide_inference" to Server class.
6. Compact the codebase to a docker image.

Todo:

Refactoring Recognition Component