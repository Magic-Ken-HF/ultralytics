import random
import shutil
from pathlib import Path

# ============ 配置区 ============
SRC_DIR = Path("C:/Users/ht/Documents/WorkSpace/calibration/data/acquisition/labels/images")          # 原始平铺目录
DST_DIR = Path("C:/Users/ht/Documents/WorkSpace/ultralytics/datasets/test")     # 输出目录
VAL_RATIO = 0.2                    # 验证集比例
SEED = 42                          # 随机种子，保证可复现
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# ================================

def main():
    random.seed(SEED)

    # 1. 收集所有图片，并检查对应标签
    images = sorted([p for p in SRC_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise SystemExit(f"在 {SRC_DIR} 中没找到图片文件")

    paired, missing_label = [], []
    for img in images:
        label = SRC_DIR / (img.stem + ".txt")
        if label.exists():
            paired.append((img, label))
        else:
            missing_label.append(img.name)

    if missing_label:
        print(f"[警告] 以下 {len(missing_label)} 张图片没有对应 .txt 标签，将被跳过：")
        for n in missing_label:
            print("  -", n)

    if not paired:
        raise SystemExit("没有任何图片与标签配对成功，终止。")

    # 2. 划分 train / val
    random.shuffle(paired)
    n_val = max(1, int(len(paired) * VAL_RATIO))
    val_set = paired[:n_val]
    train_set = paired[n_val:]

    # 3. 创建目录
    for split in ("train", "val"):
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 4. 拷贝文件
    def copy_split(pairs, split):
        for img, lbl in pairs:
            shutil.copy2(img, DST_DIR / "images" / split / img.name)
            shutil.copy2(lbl, DST_DIR / "labels" / split / lbl.name)

    copy_split(train_set, "train")
    copy_split(val_set, "val")

    print(f"\n完成：")
    print(f"  训练集: {len(train_set)} 张")
    print(f"  验证集: {len(val_set)} 张")
    print(f"  输出目录: {DST_DIR.resolve()}")

if __name__ == "__main__":
    main()