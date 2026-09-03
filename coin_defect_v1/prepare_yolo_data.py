from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载预训练模型 (yolov8n.pt 首次运行会自动下载)
    model = YOLO('yolov8n.pt')

    # 2. 启动模型训练
    results = model.train(
        data=r'C:\Users\zhongqi.sun\Desktop\coin_dataset\data.yaml',
        epochs=50,             # 先训练 50 轮
        imgsz=640,             # 图片训练尺寸
        batch=16,              # 批次大小
        workers=2,             # 加载线程数
        name='coin_defect_v1'   # 实验名称
    )