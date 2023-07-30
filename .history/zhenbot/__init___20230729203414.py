# coding=utf-8
import warnings

warnings.filterwarnings('ignore')
import io
import os
import base64
import json
import pathlib
import torch
import time
import onnxruntime
from PIL import Image, ImageChops
import numpy as np
import cv2
from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_boxes, xyxy2xywh,)


def base64_to_image(img_base64):
    img_data = base64.b64decode(img_base64)
    return Image.open(io.BytesIO(img_data))


def get_img_base64(single_image_path):
    with open(single_image_path, 'rb') as fp:
        img_base64 = base64.b64encode(fp.read())
        return img_base64.decode()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)



def img_bs64_to_det_model_input_tensor(img_base64, fp16=False):
    img_bytes = base64.b64decode(img_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    
    # read numpy array as image
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    im0 = img_cv
    # convert from BGR to RGB:
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    try:
        assert img_cv is not None
    except:
        print('Image decode Fail')
        return None, None
    im = letterbox(img_cv,640, stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im)
    im = im.half() if fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im, im0

class TypeError(Exception):
    pass


class ZhenBot(object):
    def __init__(self, ocr: bool = True, det: bool = False, slide: bool = False, beta: bool = False, use_gpu: bool = False,
                 device_id: int = 0, import_onnx_path: str = "", charsets_path: str = ""):

        self.use_import_onnx = False
        self.__word = False
        self.__resize = []
        self.__channel = 1
        if import_onnx_path != "":
            det = False
            ocr = False
            self.__graph_path = import_onnx_path
            with open(charsets_path, 'r', encoding="utf-8") as f:
                info = json.loads(f.read())
            self.__charset = info['charset']
            self.__word = info['word']
            self.__resize = info['image']
            self.__channel = info['channel']
            self.use_import_onnx = True

        if det:
            ocr = False
            self.__graph_path = os.path.join(os.path.dirname(__file__), 'common_det.onnx')
            self.__charset = []
        if ocr:
            if not beta:
                self.__graph_path = os.path.join(os.path.dirname(__file__), 'common_old.onnx')
                self.__charset = ["", "掀", "袜", "顧", "徕", "榱", "荪", "浡", "其", "炎", "玉", "恩", "劣", "徽", "廉", "桂", "拂",
                                  "設", "⒆"]
            else:
                self.__graph_path = os.path.join(os.path.dirname(__file__), 'common.onnx')
                self.__charset = ["", "笤", "谴", "膀", "荔", "佰", "电", "臁", "矍", "同", "奇", "芄", "吠", "6",
                                  "友", "唉", "怫", "荘"]
        self.slide = slide
        self.det = det
        if use_gpu:
            self.__providers = [
                ('CUDAExecutionProvider', {
                    'device_id': device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cuda_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
            ]
        else:
            self.__providers = [
                'CPUExecutionProvider',
            ]
        if ocr or det or self.use_import_onnx:
            self.__ort_session = onnxruntime.InferenceSession(self.__graph_path, providers=self.__providers)

    def preproc(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy"""
        return self.multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr)

    def get_bbox(self, image_bytes):
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        im, ratio = self.preproc(img, (416, 416))
        ort_inputs = {self.__ort_session.get_inputs()[0].name: im[None, :, :, :]}
        output = self.__ort_session.run(None, ort_inputs)
        predictions = self.demo_postprocess(output[0], (416, 416))[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio

        pred = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        try:
            final_boxes = pred[:, :4].tolist()
            result = []
            for b in final_boxes:
                if b[0] < 0:
                    x_min = 0
                else:
                    x_min = int(b[0])
                if b[1] < 0:
                    y_min = 0
                else:
                    y_min = int(b[1])
                if b[2] > img.shape[1]:
                    x_max = int(img.shape[1])
                else:
                    x_max = int(b[2])
                if b[3] > img.shape[0]:
                    y_max = int(img.shape[0])
                else:
                    y_max = int(b[3])
                result.append([x_min, y_min, x_max, y_max])
        except Exception as e:
            return []
        return result
    

    def classification(self, img):
        if self.det:
            raise TypeError("当前识别类型为目标检测")
        if not isinstance(img, (bytes, str, pathlib.PurePath, Image.Image)):
            raise TypeError("未知图片类型")
        if isinstance(img, bytes):
            image = Image.open(io.BytesIO(img))
        elif isinstance(img, Image.Image):
            image = img.copy()
        elif isinstance(img, str):
            image = base64_to_image(img)
        else:
            assert isinstance(img, pathlib.PurePath)
            image = Image.open(img)
        if not self.use_import_onnx:
            image = image.resize((int(image.size[0] * (64 / image.size[1])), 64), Image.ANTIALIAS).convert('L')
        else:
            if self.__resize[0] == -1:
                if self.__word:
                    image = image.resize((self.__resize[1], self.__resize[1]), Image.ANTIALIAS)
                else:
                    image = image.resize((int(image.size[0] * (self.__resize[1] / image.size[1])), self.__resize[1]), Image.ANTIALIAS)
            else:
                image = image.resize((self.__resize[0], self.__resize[1]), Image.ANTIALIAS)
            if self.__channel == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
        image = np.array(image).astype(np.float32)
        image = np.expand_dims(image, axis=0) / 255.
        if not self.use_import_onnx:
            image = (image - 0.5) / 0.5
        else:
            if self.__channel == 1:
                image = (image - 0.456) / 0.224
            else:
                image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                image = image[0]
                image = image.transpose((2, 0, 1))

        ort_inputs = {'input1': np.array([image]).astype(np.float32)}
        ort_outs = self.__ort_session.run(None, ort_inputs)
        result = []

        last_item = 0
        if self.__word:
            for item in ort_outs[1]:
                result.append(self.__charset[item])
        else:
            for item in ort_outs[0][0]:
                if item == last_item:
                    continue
                else:
                    last_item = item
                if item != 0:
                    result.append(self.__charset[item])

        return ''.join(result)

    def detection(self, img_bytes: bytes = None, img_base64: str = None):
        if not self.det:
            raise TypeError("当前识别类型为文字识别")
        if not img_bytes:
            img_bytes = base64.b64decode(img_base64)
        result = self.get_bbox(img_bytes)
        return result

    def get_target(self, img_bytes: bytes = None):
        image = Image.open(io.BytesIO(img_bytes))
        w, h = image.size
        starttx = 0
        startty = 0
        end_x = 0
        end_y = 0
        for x in range(w):
            for y in range(h):
                p = image.getpixel((x, y))
                if p[-1] == 0:
                    if startty != 0 and end_y == 0:
                        end_y = y

                    if starttx != 0 and end_x == 0:
                        end_x = x
                else:
                    if startty == 0:
                        startty = y
                        end_y = 0
                    else:
                        if y < startty:
                            startty = y
                            end_y = 0
            if starttx == 0 and startty != 0:
                starttx = x
            if end_y != 0:
                end_x = x
        return image.crop([starttx, startty, end_x, end_y]), starttx, startty

    def slide_match(self, target_bytes: bytes = None, background_bytes: bytes = None, simple_target: bool=False, flag: bool=False):
        if not simple_target:
            try:
                target, target_x, target_y = self.get_target(target_bytes)
                target = cv2.cvtColor(np.asarray(target), cv2.IMREAD_ANYCOLOR)
            except SystemError as e:
                # SystemError: tile cannot extend outside image
                if flag:
                    raise e
                return self.slide_match(target_bytes=target_bytes, background_bytes=background_bytes, simple_target=True, flag=True)
        else:
            target = cv2.imdecode(np.frombuffer(target_bytes, np.uint8), cv2.IMREAD_ANYCOLOR)
            target_y = 0
            target_x = 0

        background = cv2.imdecode(np.frombuffer(background_bytes, np.uint8), cv2.IMREAD_ANYCOLOR)

        background = cv2.Canny(background, 100, 200)
        target = cv2.Canny(target, 100, 200)

        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)

        res = cv2.matchTemplate(background, target, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        h, w = target.shape[:2]
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        return {"target_y": target_y,
                "target": [int(max_loc[0]), int(max_loc[1]), int(bottom_right[0]), int(bottom_right[1])]}

    def slide_comparison(self, target_bytes: bytes = None, background_bytes: bytes = None):
        target = Image.open(io.BytesIO(target_bytes)).convert("RGB")
        background = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        image = ImageChops.difference(background, target)
        background.close()
        target.close()
        image = image.point(lambda x: 255 if x > 80 else 0)
        start_y = 0
        start_x = 0
        for i in range(0, image.width):
            count = 0
            for j in range(0, image.height):
                pixel = image.getpixel((i, j))
                if pixel != (0, 0, 0):
                    count += 1
                if count >= 5 and start_y == 0:
                    start_y = j - 5

            if count >= 5:
                start_x = i + 2
                break
        return {
            "target": [start_x, start_y]
        }

    def slide_inference(
                self,
                im,
                im0,
                weights='/Users/liyaoting/Downloads/best.onnx', 
                data='/Users/liyaoting/Downloads/jomoo_slide_det_jd_1180.yaml', 
                conf_thres=0.25, 
                iou_thres=0.45, 
                classes=None, 
                agnostic_nms=False, 
                max_det=1000,  
                save_img=True,
                save_conf=True,
                save_path='bad_case_cache',
            ):

        bs = 1
        
        model = DetectMultiBackend(weights, device=torch.device('cpu'), data=data, fp16=False)

        model.warmup(imgsz=(1, 3, 640, 640))  # warmup
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        for i, det in enumerate(pred):  # per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            try:
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    det_results = []
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        det_results.append(line)
                        # ('%g ' * len(line)).rstrip() % line
                    conf_max_index = torch.tensor([result[-1] for result in det_results]).argmax()
                    if save_img and conf < torch.tensor(0.8):
                        cv2.imwrite(os.path.join(save_path, f"{'_'.join('_'.join(time.ctime().split()[1:]).split(':'))}_conf_{conf.item():.4f}.jpg"), im0)
                    return det_results[conf_max_index][1:]
                else:
                    raise Exception('No slide background detected!')
            except Exception as error:
                print(error)

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(os.path.join(save_path, f"{'_'.join('_'.join(time.ctime().split()[1:]).split(':'))}_nobox.jpg"), im0)
    

