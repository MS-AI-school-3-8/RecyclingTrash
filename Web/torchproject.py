from flask import Flask, render_template, request, Response, send_file
import numpy as np
from PIL import Image
import base64
import cv2
from ultralytics import YOLO


app = Flask(__name__)


@app.route('/')
def indexHome():
    return render_template('index.html')


@app.route('/predRecycling', methods=['POST'])
def predRecycling():
    model = YOLO("./recycling8m_best.pt")    # ...여기 있으니깐 되네... / 다른 위치에 있으면 안됨... 왠지는 몰루???  /  localhost에서는 상관 없음...

    image = request.files['up_image']
    image = Image.open(image)

    result = model.predict(image, save=False, imgsz=640, conf=0.3, device='cpu', iou=0.7)   # conf값으로 정확도 조절 가능
    boxes = result[0].boxes.xyxy
    cls = result[0].boxes.cls
    conf = result[0].boxes.conf
    cls_dict = result[0].names

    # cv2로 리사이즈 및 이미지 확인
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, c = image.shape
    image = cv2.resize(image, (640,640))

    # predict에 따른 rectangle 그리기
    for box, cls_number, conf in zip(boxes, cls, conf):
        conf_number = float(conf.item()) * 100
        conf_number = str(round(conf_number, 1)) + "%"
        cls_number_int = int(cls_number.item())
        cls_name = cls_dict[cls_number_int]
        x1, y1 ,x2 ,y2 = box
        x1_int = int(x1.item())
        y1_int = int(y1.item())
        x2_int = int(x2.item())
        y2_int = int(y2.item())
        print(x1_int, y1_int, x2_int, y2_int ,cls_name, conf_number)

        scale_factor_x = 640 / w
        scale_factor_y = 640 / h
        x1_scale = int(x1_int * scale_factor_x)
        y1_scale = int(y1_int * scale_factor_y)
        x2_scale = int(x2_int * scale_factor_x)
        y2_scale = int(y2_int * scale_factor_y)

        image = cv2.putText(image, cls_name, (x1_scale+10, y2_scale-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        image = cv2.putText(image, conf_number, (x1_scale+10, y2_scale-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        image = cv2.rectangle(image, (x1_scale, y1_scale), (x2_scale, y2_scale), (0,255,0), 2)

    # cv2.imwrite("./test_result.jpg", image)
    ret, buffer = cv2.imencode('.jpg', image)
    # img_byte_array = buffer.tobytes()
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # return Response(img_byte_array, content_type='image/jpeg')
    return render_template('index.html', return_result="success", return_img=img_base64)


@app.route('/downRecycling', methods=['GET'])
def downRecycling():
    return send_file("./static/msai8_recycling_report.pptx", as_attachment=True)    # pdf파일로 변환할 것



if __name__ == "__main__":
    app.run(host="0.0.0.0")

