import numpy as np
import matplotlib.pyplot as plt
from utils import jamm, echo

def Jam2img_clear_region_gradient(
    UAV1, UAV2, data_path,
    threshold=0.02,
    image_size=400,
    radar=(200.0, 200.0),
    radius=200.0,
    base_color=(0, 0, 255)  # 深色基准（RGB，蓝色）
):
    """
    白底 + 梯度颜色：
    - JSR < threshold 才绘制
    - JSR 越小，颜色越深
    """

    H = W = image_size
    Radar = np.array(radar, dtype=float)
    UAV1  = np.asarray(UAV1, dtype=float)
    UAV2  = np.asarray(UAV2, dtype=float)

    def img_to_math(x_img, y_img):
        return np.array([x_img, (H - 1) - y_img], dtype=float)

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    circle_mask = ((X - radar[0])**2 + (Y - radar[1])**2) <= radius**2
    ys, xs = np.where(circle_mask)

    # 白底
    out = np.ones((H, W, 3), dtype=np.float32) * 255.0

    base_color = np.array(base_color, dtype=np.float32)

    for y, x in zip(ys, xs):
        Target = img_to_math(x, y)

        Pj = jamm(Radar, Target, UAV1) + jamm(Radar, Target, UAV2)
        Pr = echo(Target, Radar)
        if Pr <= 0:
            continue

        jsr = Pj / Pr
        if jsr < threshold:
            # 归一化“未干扰程度”
            severity = 1.0 - jsr / threshold
            severity = np.clip(severity, 0.0, 1.0)

            # 白色 → base_color 的线性插值
            out[y, x, :] = (1 - severity) * 255.0 + severity * base_color
    plt.show()
    plt.imsave(data_path, out.astype(np.uint8))
    return out
def Jam2img(
    UAV1, UAV2, data_path,
    threshold=0.02,
    image_size=400,
    radar=(200.0, 200.0),
    radius=200.0,
    deep_color=(0, 0, 255)  # 最深颜色（RGB）
):
    H = W = image_size
    Radar = np.array(radar, dtype=float)
    UAV1  = np.asarray(UAV1, dtype=float)
    UAV2  = np.asarray(UAV2, dtype=float)

    def img_to_math(x_img, y_img):
        return np.array([x_img, (H - 1) - y_img], dtype=float)

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    circle_mask = ((X - radar[0])**2 + (Y - radar[1])**2) <= radius**2
    ys, xs = np.where(circle_mask)

    out = np.ones((H, W, 3), dtype=np.float32) * 255.0
    deep = np.array(deep_color, dtype=np.float32)

    for y, x in zip(ys, xs):
        Target = img_to_math(x, y)

        Pj = jamm(Radar, Target, UAV1) + jamm(Radar, Target, UAV2)
        Pr = echo(Target, Radar)
        if Pr <= 0:
            continue

        jsr = Pj / Pr
        if jsr < threshold:
            # jsr 越小 -> severity 越大 -> 颜色越深
            severity = 1.0 - jsr / threshold
            severity = np.clip(severity, 0.0, 1.0)

            # 白色(255) 与 深色(deep) 线性插值
            out[y, x, :] = (1 - severity) * 255.0 + severity * deep
    plt.show()
    plt.imsave(data_path, out.astype(np.uint8))

    return out


def Jam2img_array(
        UAV1, UAV2,
        threshold=0.02,
        image_size=400,
        radar=(200.0, 200.0),
        radius=200.0,
        deep_color=(0, 0, 255)  # RGB 格式
):
    """
    生成态势图并返回 numpy 数组，不直接保存文件。
    用于闭环仿真中的多雷达图层叠加。
    """
    H = W = image_size
    Radar = np.array(radar, dtype=float)
    UAV1 = np.asarray(UAV1, dtype=float)
    UAV2 = np.asarray(UAV2, dtype=float)

    # 图像坐标转数学坐标逻辑 (y向上)
    def img_to_math(x_img, y_img):
        return np.array([x_img, (H - 1) - y_img], dtype=float)

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    circle_mask = ((X - radar[0]) ** 2 + (Y - radar[1]) ** 2) <= radius ** 2
    ys, xs = np.where(circle_mask)

    # 初始化白底
    out = np.ones((H, W, 3), dtype=np.float32) * 255.0
    deep = np.array(deep_color, dtype=np.float32)

    for y, x in zip(ys, xs):
        Target = img_to_math(x, y)
        # 计算 JSR (干信比)
        Pj = jamm(Radar, Target, UAV1) + jamm(Radar, Target, UAV2)
        Pr = echo(Target, Radar)

        if Pr <= 0:
            continue

        jsr = Pj / Pr
        if jsr < threshold:
            # 颜色映射逻辑：JSR 越小，颜色越深
            severity = 1.0 - jsr / threshold
            severity = np.clip(severity, 0.0, 1.0)
            out[y, x, :] = (1 - severity) * 255.0 + severity * deep

    return out.astype(np.uint8)
if __name__ == "__main__":
    Radar = np.array([200.0, 200.0])
    # 你给的例子（注意：这俩位置应当被视为“数学坐标系 y向上”）
    UAV1 = np.array([100.72, 282.09])
    UAV2 = np.array([97.46, 367.81])

    Jam2img(UAV1, UAV2, "demo.png", threshold=0.02)
