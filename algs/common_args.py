import numpy as np
import queue
import pandas as pd
import os
import socket


class CommonArgs:
    def __init__(self, args):
        self.T = args["T"]  # 时间片个数
        self.delta = args["delta"]  # 每个时间片时长（600s=10min）
        self.L_max = args["L_max"]  # 也就是论文中的L*，队列最大长度
        self.L_opt = args["L_opt"]  # 队列最优长度
        self.B_max = args["B_max"]  # 电池最大容量
        self.B_start = args["B_start"]  # 初始电量
        self.K_safe = args["K_safe"]  # 安全电量系数kappa
        self.V = args["V"]  # 李雅普诺夫偏移系数
        self.Eta = args["Eta"]  # 耗电公式中 p = kf^3 的k，此处写成eta
        self.P_idle = args["P_idle"]  # 系统决策x^t=0时，拍照的耗电功率，测量得出
        self.Mu_max = args["Mu_max"]  # cpu最大处理能力
        self.Mu_min = args["Mu_min"]  # cpu最低处理能力
        self.T_s = args["T_s"]  # 每张图片利用全部设备处理能力 Mu_max 来清点人脸数所需要的时长
        self.T_r = args["T_r"]  # 每个人脸在利用全部资源 Mu_max 来识别的时间
        self.T_p = args["T_p"]  # 最快启动程序所需时间
        self.Alpha = args["Alpha"]  # Alpha 表示 \mu^t 和 P_W^t 的指数关系，默认值为3
        self.Beta = args["Beta"]  # Beta 和 Rho 表示 \mu^{\star} 和 \mu^t 的时间比值关系，Beta默认值为1,
        self.Rho = args["Rho"]  # Rho 默认值为1

        self.L_Queue = queue.Queue(maxsize=self.L_max)  # 照片队列
        self.L = np.zeros(self.T + 1, dtype=int)  # 队列内照片数
        self.Q = np.zeros(self.T + 1, dtype=int)  # 已经完成清点的照片数
        self.B = np.zeros(self.T + 1, dtype=float)  # 电池剩余量
        self.B[0] = self.B_start  # 设置初始电量
        self.Y = np.zeros(self.T + 1, dtype=int)  # 虚拟队列
        self.E = np.zeros(self.T + 1, dtype=float)  # 记录每个时刻的功耗
        self.X = np.zeros(self.T + 1, dtype=int)  # 记录每个时刻的x_t
        self.MU = np.zeros(self.T + 1, dtype=float)  # 记录每个时刻的mu_t
        self.PM = np.zeros(self.T + 1, dtype=int)  # 记录每个时刻的 p_t = m_t
        self.LosePhoto = np.zeros(self.T + 1, dtype=int)  # 丢失照片数目
        self.C = np.zeros(self.T + 1, dtype=int)  # 记录每个时刻到来照片数C_t

        self.charge_filename = args["charge_filename"]
        self.res_dir = args["res_dir"]
        self.photos_dir = args["photos_dir"]
        self.P_in = self.read_charge_file(self.charge_filename)  # 太阳能板充电功率，后期需要根据实际情况调整
        self.latest_faces = []  # 存储当前时间片清点的照片人脸数

    @staticmethod
    def read_charge_file(file_dir):
        all_csv_list = os.listdir(file_dir)  # get csv list
        all_data_frame = []
        for single_csv in all_csv_list:
            single_data_frame = pd.read_csv(os.path.join(file_dir, single_csv),
                                            sep='\t',
                                            header=0,
                                            usecols=["--Timestamp---", "Solar"],
                                            converters={"Solar": lambda x: round(int(x) / 1200, 2)},  # 60  1200
                                            skiprows=lambda x: x > 0 and x % 2 == 0
                                            )
            if single_csv == all_csv_list[0]:
                all_data_frame = single_data_frame
            else:
                all_data_frame = pd.concat([all_data_frame, single_data_frame], ignore_index=True)
        res = all_data_frame["Solar"].values
        moved_elements = res[:42]
        remaining_elements = res[42:]
        res = np.concatenate((remaining_elements, moved_elements))
        return res

    def ours_solve_xt(self, t):
        pm_list, mu_list, opt_list, E_u_t_list = [], [], [], []

        for m in range(self.L[t], 0, -1):
            N_sum = sum(list(self.L_Queue.queue)[:m])

            # 时间约束
            part_time_sum = self.T_s * m + self.T_r * N_sum + self.T_p
            if part_time_sum > self.delta:
                continue
            mu = min(max((self.Rho * self.Mu_max * part_time_sum / self.delta) ** (1 / self.Beta), self.Mu_min),
                     self.Mu_max)
            time_sum = part_time_sum * self.Rho * self.Mu_max / (mu ** self.Beta)
            if time_sum > self.delta:
                continue

            # 能量约束
            E_u_work_t = self.Eta * (mu ** self.Alpha) * time_sum
            E_u_idle_t = self.P_idle * self.delta

            E_u_t = E_u_idle_t + E_u_work_t
            E_c_t = self.P_in[t] * self.delta
            if self.B[t] + E_c_t - E_u_t <= self.K_safe * self.B_max:
                continue

            opt = self.Y[t] * min(self.L[t] - m + self.C[t], self.L_max) + self.V * E_u_work_t
            pm_list.append(m)
            mu_list.append(mu)
            opt_list.append(opt)
            E_u_t_list.append(E_u_t)

        if len(opt_list) == 0:
            return -1, 0, 0, 0
        else:
            opt_index = np.argmin(opt_list)
            return opt_list[opt_index], mu_list[opt_index], pm_list[opt_index], E_u_t_list[opt_index]

    def strict_solve_xt(self, t, m_min):
        pm_list, mu_list, E_u_t_list = [], [], []

        for m in range(self.L[t], m_min, -1):
            N_sum = sum(list(self.L_Queue.queue)[:m])

            part_time_sum = self.T_s * m + self.T_r * N_sum + self.T_p
            if part_time_sum > self.delta:
                continue
            mu = min(max((self.Rho * self.Mu_max * part_time_sum / self.delta) ** (1 / self.Beta), self.Mu_min),
                     self.Mu_max)
            time_sum = part_time_sum * self.Rho * self.Mu_max / (mu ** self.Beta)
            if time_sum > self.delta:
                continue

            E_u_work_t = self.Eta * (mu ** self.Alpha) * time_sum
            E_u_idle_t = self.P_idle * self.delta
            E_u_t = E_u_idle_t + E_u_work_t

            E_c_t = self.P_in[t] * self.delta
            if self.B[t] + E_c_t - E_u_t <= self.K_safe * self.B_max:
                continue

            pm_list.append(m)
            mu_list.append(mu)
            E_u_t_list.append(E_u_t)

        if len(pm_list) == 0:
            for m in range(m_min, 0, -1):
                N_sum = sum(list(self.L_Queue.queue)[:m])

                part_time_sum = self.T_s * m + self.T_r * N_sum + self.T_p
                mu = min(max((self.Rho * self.Mu_max * part_time_sum / self.delta) ** (1 / self.Beta), self.Mu_min),
                         self.Mu_max)
                time_sum = part_time_sum * self.Rho * self.Mu_max / (mu ** self.Beta)
                if time_sum > self.delta:
                    continue

                E_u_work_t = self.Eta * (mu ** self.Alpha) * time_sum
                E_u_idle_t = self.P_idle * self.delta
                E_u_t = E_u_idle_t + E_u_work_t
                E_c_t = self.P_in[t] * self.delta
                if self.B[t] + E_c_t - E_u_t <= self.K_safe * self.B_max:
                    continue
                return mu, m, E_u_t
            return -1, -1, -1
        else:
            opt_index = np.argmin(E_u_t_list)
            return mu_list[opt_index], pm_list[opt_index], E_u_t_list[opt_index]

    def lazy_solve_xt(self, t):
        for m in range(self.L[t], 0, -1):
            N_sum = sum(list(self.L_Queue.queue)[:m])

            part_time_sum = self.T_s * m + self.T_r * N_sum + self.T_p
            if part_time_sum > self.delta:
                continue
            mu = min(max((self.Rho * self.Mu_max / self.delta * part_time_sum) ** (1 / self.Beta), self.Mu_min),
                     self.Mu_max)
            time_sum = part_time_sum * self.Rho * self.Mu_max / (mu ** self.Beta)
            if time_sum > self.delta:
                continue

            E_u_work_t = self.Eta * (mu ** self.Alpha) * time_sum
            E_u_idle_t = self.P_idle * self.delta
            E_u_t = E_u_idle_t + E_u_work_t
            return mu, m, E_u_t

    @staticmethod
    def socket_connect(message):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('192.168.13.17', 12345)
        client_socket.connect(server_address)
        try:
            client_socket.sendall(message.encode())
            response = client_socket.recv(1024)
            data = float(response.decode())
        finally:
            client_socket.close()
        return data

    def get_real_battery(self):
        power = self.socket_connect("1")
        battery = self.B_start - power  # 由于我们购买的功耗仪电量计算规则：充电为负，放电为正，所以这里是减号
        return battery, power

    def update_args(self, t, mu_t=0, pm_t=0, alg_name=""):
        if alg_name != "active":
            for i in range(pm_t):
                self.L_Queue.get()
            for item in self.latest_faces:
                if self.L_Queue.full():
                    break
                self.L_Queue.put(item)

        self.L[t + 1] = min(self.L[t] - pm_t + self.C[t], self.L_max)
        self.LosePhoto[t] = max(self.L[t] + self.C[t] - self.L_max, 0)  # 存储t时刻丢失的照片数目

        battery, power = self.get_real_battery()
        self.B[t + 1] = min(battery + sum(self.P_in[:t + 1]) * self.delta, self.B_max)
        self.Y[t + 1] = max(self.Y[t] + self.L[t + 1] - self.L_opt, 0)
        self.E[t] = power
        self.PM[t] = pm_t
        self.MU[t] = mu_t

    def save_data(self, filename, t):
        with open(filename, "a") as f:
            np.savetxt(f,
                       np.column_stack((
                                       t, self.X[t], self.C[t], self.PM[t], self.MU[t], self.B[t], self.E[t], self.L[t],
                                       self.LosePhoto[t])),
                       delimiter=",",
                       fmt="%.3f")
