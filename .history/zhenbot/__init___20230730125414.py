# coding=utf-8
# from utils.general import (cv2, non_max_suppression, scale_boxes, xyxy2xywh,)
# from models.common import DetectMultiBackend
import cv2
import numpy as np
from PIL import Image, ImageChops
import onnxruntime
import time
import torch
from torch import nn
import torchvision
import pathlib
import json
import base64
import os
import io
import yaml
import warnings
import logging
import logging.config

warnings.filterwarnings('ignore')


def base64_to_image(img_base64):
    img_data = base64.b64decode(img_base64)
    return Image.open(io.BytesIO(img_data))


def get_img_base64(single_image_path):
    with open(single_image_path, 'rb') as fp:
        img_base64 = base64.b64encode(fp.read())
        return img_base64.decode()

LOGGING_NAME = 'ZhenBot'

def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level, }},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False, }}})
                
set_logging(LOGGING_NAME)
LOGGER = logging.getLogger(LOGGING_NAME)

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


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
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        
        fp16 = True  # FP16
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
        import onnxruntime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(w, providers=providers)
        output_names = [x.name for x in session.get_outputs()]
        meta = session.get_modelmeta().custom_metadata_map  # metadata
        if 'stride' in meta:
            stride, names = int(meta['stride']), eval(meta['names'])
        
        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        
        b, ch, h, w = im.shape  # batch, channel, height, width
        
        im = im.cpu().numpy()  # torch to numpy
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
        y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup





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
    im = letterbox(img_cv, 640, stride=32, auto=True)[0]
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
            self.__graph_path = os.path.join(
                os.path.dirname(__file__), 'common_det.onnx')
            self.__charset = []
        if ocr:
            if not beta:
                self.__graph_path = os.path.join(
                    os.path.dirname(__file__), 'common_old.onnx')
                self.__charset = ["", "掀", "袜", "顧", "徕", "榱", "荪", "浡", "其", "炎", "玉", "恩", "劣", "徽", "廉", "桂", "拂",
                                  "設", "⒆"]
            else:
                self.__graph_path = os.path.join(
                    os.path.dirname(__file__), 'common.onnx')
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
            self.__ort_session = onnxruntime.InferenceSession(
                self.__graph_path, providers=self.__providers)

    def preproc(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones(
                (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r),
                   : int(img.shape[1] * r)] = resized_img

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
                [valid_boxes[keep], valid_scores[keep, None],
                    valid_cls_inds[keep, None]], 1
            )
        return dets

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy"""
        return self.multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr)

    def get_bbox(self, image_bytes):
        img = cv2.imdecode(np.frombuffer(
            image_bytes, np.uint8), cv2.IMREAD_COLOR)

        im, ratio = self.preproc(img, (416, 416))
        ort_inputs = {self.__ort_session.get_inputs()[
            0].name: im[None, :, :, :]}
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

        pred = self.multiclass_nms(
            boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
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
            image = image.resize(
                (int(image.size[0] * (64 / image.size[1])), 64), Image.ANTIALIAS).convert('L')
        else:
            if self.__resize[0] == -1:
                if self.__word:
                    image = image.resize(
                        (self.__resize[1], self.__resize[1]), Image.ANTIALIAS)
                else:
                    image = image.resize(
                        (int(image.size[0] * (self.__resize[1] / image.size[1])), self.__resize[1]), Image.ANTIALIAS)
            else:
                image = image.resize(
                    (self.__resize[0], self.__resize[1]), Image.ANTIALIAS)
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
                image = (
                    image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
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

    def slide_match(self, target_bytes: bytes = None, background_bytes: bytes = None, simple_target: bool = False, flag: bool = False):
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
            target = cv2.imdecode(np.frombuffer(
                target_bytes, np.uint8), cv2.IMREAD_ANYCOLOR)
            target_y = 0
            target_x = 0

        background = cv2.imdecode(np.frombuffer(
            background_bytes, np.uint8), cv2.IMREAD_ANYCOLOR)

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

        # bs = 1

        # model = DetectMultiBackend(
        #     weights, device=torch.device('cpu'), data=data, fp16=False)

        # # model.warmup(imgsz=(1, 3, 640, 640))  # warmup
        # pred = model(im)
        # pred = non_max_suppression(
        #     pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
        session = ort.InferenceSession('/Users/liyaoting/Downloads/best.onnx', providers=providers)
        outname = [i.name for i in session.get_outputs()] 
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]:im.numpy()}
        outputs = session.run(outname, inp)
        output= torch.from_numpy(outputs[0])
        pred = non_max_suppression(output, conf_thres=0.2, iou_thres=0.5)
        for i, det in enumerate(pred):  # per image
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            try:
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(
                        im.shape[2:], det[:, :4], im0.shape).round()
                    det_results = []
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        det_results.append(line)
                        # ('%g ' * len(line)).rstrip() % line
                    conf_max_index = torch.tensor(
                        [result[-1] for result in det_results]).argmax()
                    if save_img and conf < torch.tensor(0.8):
                        cv2.imwrite(os.path.join(
                            save_path, f"{'_'.join('_'.join(time.ctime().split()[1:]).split(':'))}_conf_{conf.item():.4f}.jpg"), im0)
                    return det_results[conf_max_index][1:]
                else:
                    raise Exception('No slide background detected!')
            except Exception as error:
                print(error)

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(os.path.join(
                    save_path, f"{'_'.join('_'.join(time.ctime().split()[1:]).split(':'))}_nobox.jpg"), im0)
