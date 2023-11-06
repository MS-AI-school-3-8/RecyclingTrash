# https://docs.ultralytics.com/modes/predict/
# https://docs.ultralytics.com/modes/val/
import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./runs/detect/train3/weights/best.pt")

    imag_path = "./ultralytics/cfg/cell_towers_dataset/test/images/DJI_20210911151944_0097_Z_JPG_jpg.rf.b0f03d05b47cbd2b8e3fd5b336b1d3bb.jpg"
    results = model.predict(imag_path, save=False, imgsz=640, conf=0.25, device='cuda')

    for r in results:
        image_path = r.path
        boxes = r.boxes.xyxy
        cls = r.boxes.cls
        conf = r.boxes.conf    # 왜 있는지 잘 모르겠음...
        cls_dict = r.names

        # cv2로 리사이즈 및 이미지 확인
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape

        image = cv2.resize(image, (640,640))

        # predict에 따른 rectangle 그리기
        for box, cls_number, conf in zip(boxes, cls, conf):
            # conf_number = float(conf.item())    # 왜 있는지는 모르겠음...
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

            image = cv2.rectangle(image, (x1_scale, y1_scale), (x2_scale, y2_scale), (0,0,255), 2)

        # cv2.imwrite("./test.jpg", image)
        cv2.imshow("Test", image)
        cv2.waitKey(0)


