# https://docs.ultralytics.com/modes/predict/
# https://docs.ultralytics.com/modes/val/
import glob
import cv2
import random
import argparse
from ultralytics import YOLO


import io
import torch
from flask import Flask, render_template, request, Response, send_file, make_response
from werkzeug.utils import secure_filename


app = Flask(__name__)
DETECTION_URL = "/object-detection"

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if request.method != "POST": return "Wrong format"

    
    #파일 받은 후 저장
    f = request.files['file'] # postman file key 값과 동일 필요
    imagename = secure_filename(f.filename)
    f.save('./images/' + imagename)
    

    #------------------------predict---------------------------------------------------------
    model = YOLO("./pt/best.pt")
    device = torch.device('cpu')
    #test_img = glob.glob("./images/" + imagename + ".jpg") #image load

    test_img = "./images/" + imagename

    #model.predict('bus.jpg', save=True, imgsz=320, conf=0.5, device='cpu')
    results = model.predict(test_img, save=False, imgsz=640, conf=0.25, device='cpu', iou=0.7)

    #------------------------dectect area---------------------------------------------------------
    image_path = results[0].path
    boxes = results[0].boxes.xyxy
    cls = results[0].boxes.cls
    conf = results[0].boxes.conf    # 왜 있는지 잘 모르겠음...
    cls_dict = results[0].names

    # cv2로 리사이즈 및 이미지 확인
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    image = cv2.resize(image, (640,640))

    threshold = []

    # predict에 따른 rectangle 그리기
    for box, cls_number, conf in zip(boxes, cls, conf):
        conf_number = str(float(conf.item()).__round__(2))    # 확률값
        cls_number_int = int(cls_number.item())
        cls_name = cls_dict[cls_number_int]
        x1, y1 ,x2 ,y2 = box
        x1_int = int(x1.item())
        y1_int = int(y1.item())
        x2_int = int(x2.item())
        y2_int = int(y2.item())

        print(x1_int, y1_int, x2_int, y2_int ,cls_name)

        scale_factor_x = 640 / w
        scale_factor_y = 640 / h
        x1_scale = int(x1_int * scale_factor_x)
        y1_scale = int(y1_int * scale_factor_y)
        x2_scale = int(x2_int * scale_factor_x)
        y2_scale = int(y2_int * scale_factor_y)

        image = cv2.putText(image, cls_name, (x1_scale+10, y2_scale-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)     #이미지에 텍스트 넣기
        image = cv2.putText(image, conf_number, (x1_scale+10, y2_scale-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)     #이미지에 텍스트 넣기
        image = cv2.rectangle(image, (x1_scale, y1_scale), (x2_scale, y2_scale), (0,0,255), 2)

    cv2.imwrite("./results/" + imagename, image) # 결과 사진 저장
    #cv2.imshow("Test", image)
    cv2.waitKey(0)
    #--------------------------------------------------------------------------------------------

    # list, json
    #text_data = "results"
    #response = make_response(text_data)  # 탐지 결과 사진 전달
    #response.headers['Content-Type'] = 'text/plain'
    
    #이미지 파일 첨부 
    #response.set_data("./results/" + imagename)

    return send_file("./results/" + imagename, mimetype='image/jpeg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv8 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")

    opt = parser.parse_args()

    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat