import multiprocessing
from ultralytics import YOLO 

def train_model():
    model = YOLO('yolo11n-obb.pt')  # แทนที่ด้วยการเริ่มต้นโมเดลของคุณ
    data_path = r'bigbag_dataset_updated2.0/diy.yaml'
    # model.train(data=data_path, epochs=100, imgsz=640, batch=16, classes=[2], optimizer="AdamW")
    model.train(data=data_path, epochs=1000, device='cuda:0', batch=4)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # สำหรับการรองรับ Windows
    train_model()