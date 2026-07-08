import numpy as np
import matplotlib.pyplot as plt
import torch
# 定义雷达位置


def jamm(Radar, Target, UAV):
    yita = 0.313
    theta5 = 0.0175
    G_j = 5
    P_t = 1
    G_r = 40
    R = np.linalg.norm(Radar - UAV)

    vector1 = Target - Radar  # Vector CP_1
    vector2 = UAV - Radar  # Vector CP_2

    # Compute dot product
    dotProduct = np.dot(vector1, vector2)

    # Compute magnitudes of vectors
    magnitude1 = np.linalg.norm(vector1)  # |CP_1|
    magnitude2 = np.linalg.norm(vector2)  # |CP_2|

    # Compute the angle (in radians)
    theta = np.arccos(dotProduct / (magnitude1 * magnitude2))

    if 0 <= theta < theta5 / 2:
        G_r1 = G_r
    elif theta5 / 2 <= theta < np.pi / 2:
        G_r1 = yita * (theta5 / theta)**2 * G_r
    else:
        G_r1 = yita * (2 * theta5 / np.pi)**2 * G_r

    P_j = P_t * G_j * G_r1 / (4 * np.pi)**2 / R**2
    return P_j

def echo(Target, Radar):
    delta = 25
    P_ts = 10
    G_r = 40
    R_h = np.linalg.norm(Radar - Target)
    Q_h = P_ts * G_r / (4 * np.pi) / R_h**2
    P_r = Q_h * G_r * delta / (4 * np.pi)**2 / R_h**2
    return P_r

# 随机生成一个满足距离条件的 UAV 位置
def generate_UAV_position(Radar, max_distance=600):
    # 随机生成与雷达的距离（范围为 [0, max_distance]）
    distance = np.random.uniform(0, max_distance)

    # 随机生成角度，范围是 [0, 2*pi]
    angle = np.random.uniform(0, 2 * np.pi)  # 角度范围是 0 到 2 * pi

    # 计算 UAV 位置（极坐标转笛卡尔坐标）
    x_offset = distance * np.cos(angle)
    y_offset = distance * np.sin(angle)

    UAV_position = Radar + np.array([x_offset, y_offset])
    return UAV_position, angle, distance

def generate_two_UAV_positions(Radar, min_angle=30, max_angle=90, min_distance=50, max_distance=400, max_dist_diff=200):
    # UAV1
    angle1 = np.random.uniform(0, 2 * np.pi)
    dist1 = np.random.uniform(min_distance, max_distance)
    x1 = Radar[0] + dist1 * np.cos(angle1)
    y1 = Radar[1] + dist1 * np.sin(angle1)
    UAV1 = np.array([x1, y1])

    # 偏移角度：随机一个夹角在 [min_angle, max_angle]，然后加/减到原始角度
    angle_offset = np.deg2rad(np.random.uniform(min_angle, max_angle))
    if np.random.rand() < 0.5:
        angle_offset = -angle_offset  # 随机左右旋转

    angle2 = angle1 + angle_offset

    # UAV2 距离范围：在 [max(100, dist1 - 200), min(400, dist1 + 200)]
    min_dist2 = max(min_distance, dist1 - max_dist_diff)
    max_dist2 = min(max_distance, dist1 + max_dist_diff)
    dist2 = np.random.uniform(min_dist2, max_dist2)

    x2 = Radar[0] + dist2 * np.cos(angle2)
    y2 = Radar[1] + dist2 * np.sin(angle2)
    UAV2 = np.array([x2, y2])

    return UAV1, UAV2




# 可视化生成的 UAV 位置
def plot_UAV_positions(Radar, UAV1, UAV2):
    plt.figure(figsize=(6, 6))
    plt.scatter(Radar[0], Radar[1], c='red', label='Radar', s=100)
    plt.scatter(UAV1[0], UAV1[1], c='blue', label='UAV1', s=50)
    plt.scatter(UAV2[0], UAV2[1], c='green', label='UAV2', s=50)

    # plt.xlim(0, 1000)
    # plt.ylim(0, 1000)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def sort_target(target):
    """
    按照 x 坐标排序 target，使其满足 x1 < x2。
    target 形状: [batch_size, 4]，格式为 [x1, y1, x2, y2]。
    """
    x1, y1, x2, y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # 需要交换的索引：当 x1 > x2 时
    swap_mask = x1 > x2

    # 交换 x1, y1 和 x2, y2
    x1_new = torch.where(swap_mask, x2, x1)
    y1_new = torch.where(swap_mask, y2, y1)
    x2_new = torch.where(swap_mask, x1, x2)
    y2_new = torch.where(swap_mask, y1, y2)

    return torch.stack([x1_new, y1_new, x2_new, y2_new], dim=1)


# 主程序
if __name__ == "__main__":
    Radar = np.array([200, 200])
    UAV1, UAV2 = generate_two_UAV_positions(Radar)
    print("UAV1 Position:", UAV1)
    print("UAV2 Position:", UAV2)
    plot_UAV_positions(Radar, UAV1, UAV2)
