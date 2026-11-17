from ultralytics import YOLO

def main():
    # 加载模型
    #model = YOLO("yolo26n.yaml")  # 从 YAML 构建新模型
    model = YOLO("yolo26m.pt")  # 加载预训练模型（推荐用于训练）
    #model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # 从 YAML 构建模型并迁移权重

    # 训练模型
    results = model.train(
        data="coco8.yaml",
        epochs=100,
        imgsz=1024,
        device=0,     # 显式指定使用 GPU 0
        workers=0     # Windows 下禁用多进程 DataLoader，避免 worker 崩溃
    )

if __name__ == '__main__':
    main()