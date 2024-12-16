import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from utils.math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import copy


class UAVEnv:
    def __init__(self, length=2, num_obstacle=3, num_agents=4):
        self.length = length  # 环境大小：2 x 2
        self.num_obstacle = num_obstacle  # 障碍物个数
        self.num_agents = num_agents  # 无人机个数，包括目标无人机
        self.time_step = 0.5  # update time step
        self.v_max = 0.1  # 最大速度
        self.v_max_e = 0.12  # 最大速度
        self.a_max = 0.04  # 最大加速度
        self.a_max_e = 0.05  # 最大加速度
        self.L_sensor = 0.2
        self.num_lasers = 16  # 激光束数量
        self.multi_current_lasers = [
            [self.L_sensor for _ in range(self.num_lasers)]
            for _ in range(self.num_agents)
        ]  # 每一个无人机的激光束长度
        self.agents = ["agent_0", "agent_1", "agent_2", "target"]
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]  # 障碍物
        self.history_positions = [
            [] for _ in range(num_agents)
        ]  # 记录无人机的历史位置, 用于绘图

        self.action_space = {
            "agent_0": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "agent_1": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "agent_2": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        }  # 动作空间，表示 xy 方向的加速度
        self.observation_space = {
            "agent_0": spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            "agent_1": spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            "agent_2": spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(23,)),
        }  # 观测空间，包括自身空间(4 维，速度位置)；队友空间(4 位置)；激光束(16 维)；目标空间(2 维，距离角度)

    def reset(self):
        """
        1. 初始化无人机位置
        2. 更新激光束信息
        3. 获取所有无人机的观察信息
        return: 无人机的观察信息
        """
        self.multi_current_pos = []  # 无人机当前位置
        self.multi_current_vel = []  # 无人机当前速度
        self.history_positions = [
            [] for _ in range(self.num_agents)
        ]  # 记录无人机的历史位置, 用于绘图

        # 初始化无人机位置
        for i in range(self.num_agents):
            if i != self.num_agents - 1:  # 对于非目标无人机
                self.multi_current_pos.append(
                    np.random.uniform(low=0.1, high=0.4, size=(2,))
                )
            else:  # 对于目标无人机
                self.multi_current_pos.append(np.array([0.5, 1.75]))

            self.multi_current_vel.append(np.zeros(2))  # 初始化所有无人机速度为 0

        # 更新激光束信息
        self.update_lasers_isCollied_wrapper()
        # 获取所有无人机的观察信息
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self, actions):
        """
        1. 更新无人机速度、位置
        2. 更新障碍物位置
        3. 更新激光束信息并判断是否碰撞
        4. 计算奖励和任务完成状态
        return: 下一步的观察信息、奖励、任务完成状态
        """
        last_d2target = []

        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]  # 获取无人机位置
            if i != self.num_agents - 1:  # 非目标无人机
                pos_taget = self.multi_current_pos[-1]  # 获取目标无人机位置
                last_d2target.append(
                    np.linalg.norm(pos - pos_taget)
                )  # 记录上一步无人机到目标的距离

            self.multi_current_vel[i][0] += actions[i][0] * self.time_step  # 更新速度
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step  # 更新速度
            vel_magnitude = np.linalg.norm(self.multi_current_vel)  # 计算速度大小
            if i != self.num_agents - 1:  # 非目标无人机
                if vel_magnitude >= self.v_max:  # 如果速度大于最大速度，进行归一化缩放
                    self.multi_current_vel[i] = (
                        self.multi_current_vel[i] / vel_magnitude * self.v_max
                    )
            else:
                if (
                    vel_magnitude >= self.v_max_e
                ):  # 如果速度大于最大速度，进行归一化缩放
                    self.multi_current_vel[i] = (
                        self.multi_current_vel[i] / vel_magnitude * self.v_max_e
                    )

            self.multi_current_pos[i][0] += (
                self.multi_current_vel[i][0] * self.time_step
            )  # 更新位置
            self.multi_current_pos[i][1] += (
                self.multi_current_vel[i][1] * self.time_step
            )  # 更新位置

        # 更新障碍物位置
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step  # 更新障碍物位置
            for dim in [
                0,
                1,
            ]:  # 如果障碍物越界，则反转其速度，使其返回环境范围内。
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        Collided = self.update_lasers_isCollied_wrapper()  # 判断是否碰撞
        rewards, dones = self.cal_rewards_dones(
            Collided, last_d2target
        )  # 计算奖励和任务完成状态
        multi_next_obs = self.get_multi_obs()  # 获取所有无人机观测信息

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
        """
        1. 获取无人机的位置和速度
        2. 归一化处理
        return: 所有无人机的观测信息
        """
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]  # 获取无人机位置
            vel = self.multi_current_vel[i]  # 获取无人机速度
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max,
            ]  # 进行归一化处理
            total_obs.append(S_uavi)
        return total_obs

    def get_multi_obs(self):
        """
        1. 获取无人机的位置、速度，归一化处理
        return: [无人机自身速度位置、队友位置、激光信息、目标距离]
        return: [无人机自身速度位置、激光信息、目标距离]
        """
        total_obs = []  # 存储所有无人机的观测信息
        single_obs = []  # 存储单个无人机的观测信息
        S_evade_d = []  # 针对目标无人机，存储目标的逃避距离
        for i in range(self.num_agents):  # 遍历所有无人机
            pos = self.multi_current_pos[i]  # 获取无人机位置
            vel = self.multi_current_vel[i]  # 获取无人机速度
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max,
            ]  # 进行归一化处理
            S_team = []  # 存储其他智能体的位置，包括 3 个无人机和 1 个目标无人机
            S_target = []  #  存储目标的位置和目标与当前智能体的距离、角度等信息
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1:  # 非目标无人机且非当前无人机
                    pos_other = self.multi_current_pos[j]  # 获取其他无人机的位置
                    S_team.extend(
                        [pos_other[0] / self.length, pos_other[1] / self.length]
                    )  # 归一化处理
                elif j == self.num_agents - 1:  # 目标无人机
                    pos_target = self.multi_current_pos[j]  # 获取目标无人机的位置
                    d = np.linalg.norm(
                        pos - pos_target
                    )  # 计算目标无人机与当前无人机的距离
                    theta = np.arctan2(
                        pos_target[1] - pos[1], pos_target[0] - pos[0]
                    )  # 计算目标无人机与当前无人机的角度
                    S_target.extend(
                        [d / np.linalg.norm(2 * self.length), theta]
                    )  # 归一化处理
                    if i != self.num_agents - 1:  # 针对目标无人机，计算逃避距离
                        S_evade_d.append(d / np.linalg.norm(2 * self.length))

            S_obser = self.multi_current_lasers[i]  # 获取当前无人机的激光信息

            if i != self.num_agents - 1:  # 非目标无人机
                single_obs = [
                    S_uavi,
                    S_team,
                    S_obser,
                    S_target,
                ]  # 4 + 2 x 2 + 16 + 2 = 26
            else:  # 目标无人机
                single_obs = [S_uavi, S_obser, S_evade_d]  # 4 + 16 + 3 = 23
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)

        return total_obs

    def cal_rewards_dones(self, IsCollied, last_d):
        """
        1. 接近奖励、安全奖励、多阶段奖励、完成奖励
        return: 奖励、任务完成状态
        """
        dones = [False] * self.num_agents  # 初始化所有智能体的任务完成状态为False
        rewards = np.zeros(self.num_agents)  # 初始化所有智能体的奖励为 0
        mu1 = 0.7  # 逼近奖励系数
        mu2 = 0.4  # 安全奖励系数，避免碰撞
        mu3 = 0.01  # 多阶段奖励
        mu4 = 5  # 完成奖励
        d_capture = 0.3  # 捕获距离
        d_limit = 0.75  # 跟踪距离

        # 单个智能体接近目标的奖励
        for i in range(3):
            pos = self.multi_current_pos[i]  # 当前无人机位置
            vel = self.multi_current_vel[i]  # 当前无人机速度
            pos_target = self.multi_current_pos[-1]  # 目标无人机位置
            v_i = np.linalg.norm(vel)  # 当前智能体的速度大小
            dire_vec = pos_target - pos  # 目标方向向量
            d = np.linalg.norm(dire_vec)  # 当前智能体到目标的距离

            cos_v_d = np.dot(vel, dire_vec) / (
                v_i * d + 1e-3
            )  # 计算速度方向和目标方向的余弦值
            r_near = abs(2 * v_i / self.v_max) * cos_v_d  # 接近目标的奖励
            rewards[i] += mu1 * r_near  # 根据接近目标奖励更新奖励

        # 碰撞惩罚
        for i in range(self.num_agents):
            if IsCollied[i]:  # 如果智能体发生碰撞
                r_safe = -10  # 设置较大的惩罚
            else:
                lasers = self.multi_current_lasers[i]  # 当前智能体的激光雷达测量
                r_safe = (
                    min(lasers) - self.L_sensor - 0.1
                ) / self.L_sensor  # 安全奖励，根据激光雷达距离计算
            rewards[i] += mu2 * r_safe  # 根据安全奖励更新奖励

        # 多阶段奖励
        p0 = self.multi_current_pos[0]  # 无人机 0 位置
        p1 = self.multi_current_pos[1]  # 无人机 1 位置
        p2 = self.multi_current_pos[2]  # 无人机 2 位置
        pe = self.multi_current_pos[-1]  # 目标无人机位置
        S1 = cal_triangle_S(p0, p1, pe)  # 计算三角形面积
        S2 = cal_triangle_S(p1, p2, pe)  # 计算三角形面积
        S3 = cal_triangle_S(p2, p0, pe)  # 计算三角形面积
        S4 = cal_triangle_S(p0, p1, p2)  # 计算三角形面积
        d1 = np.linalg.norm(p0 - pe)  # 计算无人机 0 到目标的距禙
        d2 = np.linalg.norm(p1 - pe)  # 计算无人机 1 到目标的距禙
        d3 = np.linalg.norm(p2 - pe)  # 计算无人机 2 到目标的距禙
        Sum_S = S1 + S2 + S3  # 计算三角形面积和
        Sum_d = d1 + d2 + d3  # 计算距离和
        Sum_last_d = sum(last_d)  # 计算上一步距离和

        # 目标奖励
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d), -2, 2)  # 鼓励目标的距离减少
        # 跟踪奖励
        if (
            Sum_S > S4
            and Sum_d >= d_limit
            and all(d >= d_capture for d in [d1, d2, d3])
        ):  # 如果三角形面积和大于 S4，且距离大于 d_limit，且所有无人机到目标的距离大于 d_capture
            r_track = -Sum_d / max([d1, d2, d3])
            rewards[0:2] += mu3 * r_track
        # 包围奖励
        elif Sum_S > S4 and (
            Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])
        ):  # 如果三角形面积和大于 S4，且距离小于 d_limit，或者有无人机到目标的距离大于 d_capture
            r_encircle = -1 / 3 * np.log(Sum_S - S4 + 1)
            rewards[0:2] += mu3 * r_encircle
        # 捕获奖励
        elif Sum_S == S4 and any(
            d > d_capture for d in [d1, d2, d3]
        ):  # 如果三角形面积和等于 S4，且有无人机到目标的距离大于 d_capture
            r_capture = np.exp((Sum_last_d - Sum_d) / (3 * self.v_max))
            rewards[0:2] += mu3 * r_capture

        # 完成奖励
        if Sum_S == S4 and all(d <= d_capture for d in [d1, d2, d3]):
            rewards[0:2] += mu4 * 10  # 完成任务的奖励
            dones = [True] * self.num_agents  # 设置所有智能体的任务完成状态为 True

        return rewards, dones

    def update_lasers_isCollied_wrapper(self):
        """
        1. 更新激光信息并判断是否碰撞
        return: 是否碰撞
        """
        self.multi_current_lasers = []  # 存储每个无人机的当前激光信息
        dones = []  # 存储每个无人机是否碰撞
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]  # 当前无人机位置
            current_lasers = [
                self.L_sensor
            ] * self.num_lasers  # 初始化当前无人机的激光距离为传感器最大距离
            done_obs = []
            for obs in self.obstacles:  # 遍历所有障碍物
                obs_pos = obs.position  # 障碍物位置
                r = obs.radius  # 障碍物半径
                _current_lasers, done = update_lasers(
                    pos, obs_pos, r, self.L_sensor, self.num_lasers, self.length
                )  # 更新激光信息并记录无人机是否与障碍物发生碰撞
                current_lasers = [
                    min(l, cl) for l, cl in zip(_current_lasers, current_lasers)
                ]  # 取最大激光距离和已经计算的最小激光值，表示当前激光的有效距离
                done_obs.append(done)  # 记录是否碰撞
            done = any(done_obs)  # 如果与其中一个障碍物发生碰撞，则标记为碰撞
            if done:  # 如果发生碰撞，设置智能体速度为零
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)  # 记录当前无人机的激光信息
            dones.append(done)  # 记录是否碰撞
        return dones

    def render(self):
        """
        绘图
        """
        plt.clf()

        uav_icon = mpimg.imread("envs/src/uav.png")  # 加载无人机图标

        for i in range(self.num_agents - 1):  # 遍历非目标无人机
            pos = copy.deepcopy(self.multi_current_pos[i])  # 获取无人机位置
            vel = self.multi_current_vel[i]  # 获取无人机速度
            self.history_positions[i].append(pos)  # 记录无人机历史位置
            trajectory = np.array(self.history_positions[i])
            plt.plot(
                trajectory[:, 0], trajectory[:, 1], "b-", alpha=0.3
            )  # 绘制无人机轨迹
            angle = np.arctan2(vel[1], vel[0])  # 根据当前速度计算无人机速度方向
            t = (
                transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            )  # 旋转无人机图标以表示速度方向
            icon_size = 0.1  # 图标大小
            plt.imshow(
                uav_icon,
                transform=t + plt.gca().transData,
                extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2),
            )

        # 绘制目标无人机的位置以及轨迹
        plt.scatter(
            self.multi_current_pos[-1][0],
            self.multi_current_pos[-1][1],
            c="r",
            label="Target",
        )
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], "r-", alpha=0.3)

        # 绘制障碍物的位置
        for obstacle in self.obstacles:
            circle = plt.Circle(
                obstacle.position, obstacle.radius, color="gray", alpha=0.5
            )
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()
        plt.legend()

        # 保存当前图形并返回图像
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()

        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        """
        绘图
        """
        plt.clf()

        uav_icon = mpimg.imread("envs/src/uav.png")

        # 绘制无人机位置
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)

            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))
                plt.plot(
                    trajectory[j : j + 2, 0],
                    trajectory[j : j + 2, 1],
                    color=color,
                    alpha=0.7,
                )

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(
                uav_icon,
                transform=t + plt.gca().transData,
                extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2),
            )

        # 绘制无人机轨迹
        plt.scatter(
            self.multi_current_pos[-1][0],
            self.multi_current_pos[-1][1],
            c="r",
            label="Target",
        )
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], "r-", alpha=0.3)

        # 绘制障碍物的位置
        for obstacle in self.obstacles:
            circle = plt.Circle(
                obstacle.position, obstacle.radius, color="gray", alpha=0.5
            )
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    def close(self):
        plt.close()


class obstacle:
    def __init__(self, length=2):  # 障碍物初始化, 随机生成障碍物位置、速度、半径
        self.position = np.random.uniform(
            low=0.45, high=length - 0.55, size=(2,)
        )  # 障碍物位置
        angle = np.random.uniform(0, 2 * np.pi)  # 障碍物运动方向
        speed = 0.03  # 障碍物速度
        self.velocity = np.array(
            [speed * np.cos(angle), speed * np.sin(angle)]
        )  # 障碍物在两个方向上的速度
        self.radius = np.random.uniform(0.1, 0.15)  # 障碍物半径
