# encoding=utf-8
import argparse
import base64
import json

import zhenbot
from flask import Flask, request

parser = argparse.ArgumentParser(description="recognize-detect-server")
parser.add_argument("-p", "--port", type=int, default=9898)
parser.add_argument("--ocr", action="store_true", help="activate char-recognize model")
parser.add_argument("--det", action="store_true", help="activate object-detect model")
parser.add_argument('--slide', action='store_true', help='activate slide-inference model')

args = parser.parse_args()

app = Flask(__name__)


class Server(object):
    def __init__(self, ocr=False, det=False, slide=True):
        self.ocr_option = ocr
        self.det_option = det
        self.slide_option = slide
        self.ocr = None
        self.det = None
        self.slide = None
        if self.ocr_option:
            print("Char-Recognize Server Start!")
            self.ocr = zhenbot.ZhenBot()
        else:
            print("Char-Recognize Server Stop! To start with parameter  --ocr")
        if self.det_option:
            print("Object-Detect Server Start!")
            self.det = zhenbot.ZhenBot(det=True)
        else:
            print("Object-Detect Server Stop! To start with parameter  --det")
        if self.slide_option:
            print("Slide-Inference Server Start!")
            self.slide = zhenbot.ZhenBot(slide=True)
        else:
            print("Object-Detect Server Stop! To start with parameter  --det")

    def classification(self, img: bytes):
        if self.ocr_option:
            return self.ocr.classification(img)
        else:
            raise Exception("char-recognize model unuse")

    def detection(self, img: bytes):
        if self.det_option:
            return self.det.detection(img)
        else:
            raise Exception("object-detect model unuse")

    # def slide(self, target_img: bytes, bg_img: bytes, algo_type: str):
    #     dddd = self.ocr or self.det or zhenbot.ZhenBot(ocr=False)
    #     if algo_type == 'match':
    #         return dddd.slide_match(target_img, bg_img)
    #     elif algo_type == 'compare':
    #         return dddd.slide_comparison(target_img, bg_img)
    #     else:
    #         raise Exception(f"不支持的滑块算法类型: {algo_type}")

    def slide_inference(self, im, im0):
        if self.slide_option:
            return self.slide.slide_inference(im, im0)
        else:
            raise Exception("slide-inference model unuse")

server = Server(ocr=args.ocr, det=args.det, slide=args.slide)


def get_img(request, img_type='file', img_name='image'):
    if img_type == 'b64':
        img = base64.b64decode(request.get_data()) # 
        try: # json str of multiple images
            dic = json.loads(img)
            img = base64.b64decode(dic.get(img_name).encode())
        except Exception as e: # just base64 of single image
            pass
    else:
        img = request.files.get(img_name).read()
    return img

def get_img_for_slide(request, img_type='b64', img_name='image'):
    if img_type == 'b64':
        try: # json str of multiple images
            dic = json.loads(request.get_data())
            img_bytes = base64.b64decode(dic.get(img_name))
            im, im0 = zhenbot.img_bs64_to_det_model_input_tensor(img_bytes, input_type='img_bytes')
        except Exception as e: # just base64 of single image
            pass
    else:
        # img = request.files.get(img_name).read()
        raise TypeError('Slide-inference model only support base64 input in version 0.1')
    return im, im0

def set_ret(result, ret_type='text'):
    if ret_type == 'json':
        if isinstance(result, Exception):
            return json.dumps({"status": 200, "result": "", "msg": str(result)})
        else:
            return json.dumps({"status": 200, "result": result, "msg": ""})
        # return json.dumps({"succ": isinstance(result, str), "result": str(result)})
    else:
        if isinstance(result, Exception):
            return ''
        else:
            return str(result).strip()


@app.route('/<opt>/<img_type>', methods=['POST'])
@app.route('/<opt>/<img_type>/<ret_type>', methods=['POST'])
def ocr(opt, img_type='file', ret_type='text'):
    try:
        img = get_img(request, img_type)
        if opt == 'ocr':
            result = server.classification(img)
        elif opt == 'det':
            result = server.detection(img)
        elif opt == 'slide':
            im, im0 = get_img_for_slide(request)
            result = server.slide_inference(im, im0)
            result += f' {im0.shape[0]} {im0.shape[1]}'
        else:
            raise f"<opt={opt}> is invalid"
        return set_ret(result, ret_type)
    except Exception as e:
        print(e)
        return set_ret(e, ret_type)

# @app.route('/slide/<algo_type>/<img_type>', methods=['POST'])
# @app.route('/slide/<algo_type>/<img_type>/<ret_type>', methods=['POST'])
# def slide(algo_type='compare', img_type='file', ret_type='text'):
#     try:
#         target_img = get_img(request, img_type, 'target_img')
#         bg_img = get_img(request, img_type, 'bg_img')
#         result = server.slide(target_img, bg_img, algo_type)
#         return set_ret(result, ret_type)
#     except Exception as e:
#         return set_ret(e, ret_type)

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=args.port)