import numpy as np
import math


def update_lasers(pos, obs_pos, r, L, num_lasers, bound):
    """
    1. 计算无人机与障碍物的距离
    2. 如果无人机与障碍物发生碰撞，返回 0
    return: 更新后的激光长度，是否发生碰撞
    """
    distance_to_obs = np.linalg.norm(
        np.array(pos) - np.array(obs_pos)
    )  # 计算无人机与障碍物的距离
    isInObs = (
        distance_to_obs < r
        or pos[0] < 0
        or pos[0] > bound
        or pos[1] < 0
        or pos[1] > bound
    )  # 判断无人机是否与障碍物发生碰撞

    if isInObs:  # 如果无人机与障碍物发生碰撞，返回 0
        return [0.0] * num_lasers, isInObs

    angles = np.linspace(
        0, 2 * np.pi, num_lasers, endpoint=False
    )  # 计算每个激光束的角度
    laser_lengths = [L] * num_lasers  # 初始化每个激光束的长度

    for i, angle in enumerate(
        angles
    ):  # 遍历每个激光传感器，根据障碍物与边界的情况更新激光的长度
        intersection_dist = check_obs_intersection(
            pos, angle, obs_pos, r, L
        )  # 计算激光与障碍物的交点距离
        if laser_lengths[i] > intersection_dist:  # 更新激光的实际探测距离
            laser_lengths[i] = intersection_dist

        wall_dist = check_wall_intersection(
            pos, angle, bound, L
        )  # 计算激光与边界的交点距离
        if laser_lengths[i] > wall_dist:  # 更新激光的实际探测距离
            laser_lengths[i] = wall_dist

    return laser_lengths, isInObs


def check_obs_intersection(start_pos, angle, obs_pos, r, max_distance):
    ox = obs_pos[0]
    oy = obs_pos[1]

    end_x = start_pos[0] + max_distance * np.cos(angle)
    end_y = start_pos[1] + max_distance * np.sin(angle)

    dx = end_x - start_pos[0]
    dy = end_y - start_pos[1]
    fx = start_pos[0] - ox
    fy = start_pos[1] - oy

    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = (fx**2 + fy**2) - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        if 0 <= t1 <= 1:
            return t1 * max_distance
        if 0 <= t2 <= 1:
            return t2 * max_distance

    return max_distance


def check_wall_intersection(start_pos, angle, bound, L):

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    L_ = L

    if sin_theta > 0:
        L_ = min(L_, abs((bound - start_pos[1]) / sin_theta))

    if sin_theta < 0:
        L_ = min(L_, abs(start_pos[1] / -sin_theta))

    if cos_theta > 0:
        L_ = min(L_, abs((bound - start_pos[0]) / cos_theta))

    if cos_theta < 0:
        L_ = min(L_, abs(start_pos[0] / -cos_theta))

    return L_


def cal_triangle_S(p1, p2, p3):
    S = abs(
        0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    )
    if math.isclose(S, 0.0, abs_tol=1e-9):
        return 0.0
    else:
        return S
