import pandas as pd
from pathlib import Path

# 原始 CSV 路径
old_csv = Path("re_train.csv")  # 你原来的 CSV
new_csv = Path("re_train_paired.csv")  # 生成新的 CSV

# 加载原始 CSV
df = pd.read_csv(old_csv)

# 提取文件名，例如 image_0.png
df['filename'] = df['image_path'].apply(lambda p: Path(p).name)

# 拼接新的全路径（带JAM_data/train前缀）
df['re_image'] = df['filename'].apply(lambda name: f"JAM_data/train/re_images/{name}")
df['image'] = df['filename'].apply(lambda name: f"JAM_data/train/images/{name}")

# 重建 DataFrame，列顺序为：re_image, image, x, y, a, b
df_new = df[['re_image', 'image', 'x', 'y', 'a', 'b']]

# 保存新 CSV
df_new.to_csv(new_csv, index=False)

print(f"新 CSV 已保存至: {new_csv}")
