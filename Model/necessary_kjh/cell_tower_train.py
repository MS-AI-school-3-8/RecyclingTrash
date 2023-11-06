# https://docs.ultralytics.com/modes/train/
# https://inhovation97.tistory.com/32
from ultralytics import YOLO
import time

if __name__ == "__main__":

    def train_model(epochs, period, batch, model_path, opt, lr0):
        model = YOLO(model_path)
        model.train(data="./cell_towers.yaml", epochs=epochs, save_period=period, patience=50, batch=batch, lr0=lr0, lrf=0.01, 
                    device="cuda", optimizer=opt, name=model_path[7:9] + "_" + opt + "_" + str(lr0)[2:])

    start_time = time.time()
    # batch는 32 or 64
    train_model(epochs=100, period=10, batch=32, model_path="./yolov8m.pt", opt="SGD", lr0=0.05)
    train_model(epochs=100, period=10, batch=32, model_path="./yolov8m.pt", opt="SGD", lr0=0.01)
    train_model(epochs=100, period=10, batch=32, model_path="./yolov8m.pt", opt="Adam", lr0=0.005)
    train_model(epochs=100, period=10, batch=32, model_path="./yolov8m.pt", opt="Adam", lr0=0.001)
    print("total time :", time.time() - start_time)
