import numpy as np
import threading
import cv2
import os
import sys
import json
import time
import datetime
import random
import subprocess
import common_args


class Ours:
    def __init__(self, all_args):
        self.all_args = all_args

    def alg_iterate(self, t):
        global start_index

        start_time = time.time()

        v_list = [0, 0]  # 存储x_t分别为0/1时的最优值
        v_list[0] = self.all_args.Y[t] * min(self.all_args.L[t] + self.all_args.C[t], self.all_args.L_max)

        mu_t, pm_t, E_u_t = 0, 0, 0
        if self.all_args.Y[t] == 0:
            self.all_args.X[t] = 0
        else:
            v_list[1], mu_t, pm_t, E_u_t = self.all_args.ours_solve_xt(t)
            if v_list[1] > 0:
                self.all_args.X[t] = np.argmin(v_list)

        # 执行决策，并更新参数
        during_time = int(time.time() - start_time)
        if self.all_args.X[t] == 1:
            need_processed_faces = list(self.all_args.L_Queue.queue)[:pm_t]
            real_pm = decide_execute(self.all_args.delta - during_time, mu_t, pm_t, need_processed_faces,
                                     self.all_args.photos_dir)
            start_index += real_pm

            self.all_args.update_args(t, mu_t=mu_t, pm_t=real_pm)  # 更新参数
            during_time = time.time() - start_time
            sleep_time = self.all_args.delta - during_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"决策执行超时{sleep_time}秒")
        else:
            self.all_args.update_args(t)
            sleep_time = self.all_args.delta - during_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"决策执行超时{sleep_time}秒")

    def run(self, res_filename):
        global last_index
        global latest_index

        with open(res_filename, "a+") as f:
            f.write(str(datetime.datetime.now()) + "\n")
            f.write("t, X, C, PM, MU, B, E, L, lose_photo\n")

        for t in range(self.all_args.T):
            now_latest_index = latest_index
            self.all_args.C[t] = now_latest_index - last_index
            self.all_args.latest_faces = count_faces(self.all_args.photos_dir, now_latest_index)
            last_index = now_latest_index
            if self.all_args.B[t] > self.all_args.K_safe * self.all_args.B_max:
                self.alg_iterate(t)
            else:
                time.sleep(self.all_args.delta)
            self.all_args.save_data(res_filename, t)


class Strict:
    def __init__(self, all_args):
        self.all_args = all_args

    def alg_iterate(self, t):
        global start_index

        start_time = time.time()

        m_min = int(self.all_args.L[t] + self.all_args.C[t] - (t + 1) * self.all_args.L_opt + sum(self.all_args.L[:t]))
        m_min = min(m_min, self.all_args.L[t])

        mu_t, pm_t, E_u_t = 0, 0, 0
        if m_min > 0:
            self.all_args.X[t] = 1
            mu_t, pm_t, E_u_t = self.all_args.strict_solve_xt(t, m_min)

        # 执行决策，并更新参数
        during_time = int(time.time() - start_time)
        if self.all_args.X[t] == 1:
            need_processed_faces = list(self.all_args.L_Queue.queue)[:pm_t]
            real_pm = decide_execute(self.all_args.delta - during_time, mu_t, pm_t, need_processed_faces,
                                     self.all_args.photos_dir)
            start_index += real_pm  # 更新照片队列起始index

            self.all_args.update_args(t, mu_t=mu_t, pm_t=real_pm)

            during_time = time.time() - start_time
            sleep_time = self.all_args.delta - during_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"决策执行超时{sleep_time}秒")
        else:
            self.all_args.update_args(t)
            sleep_time = self.all_args.delta - during_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"决策执行超时{sleep_time}秒")

    def run(self, res_filename):
        global last_index
        global latest_index
        global start_index

        with open(res_filename, "a") as f:
            f.write("\n" + str(datetime.datetime.now()) + "\n")
            f.write("t, X, C, PM, MU, B, E, L, lose_photo\n")

        for t in range(self.all_args.T):
            self.all_args.C[t] = latest_index - last_index
            if self.all_args.L[t] + self.all_args.C[t] > self.all_args.L_max:
                latest_index = start_index + self.all_args.L_max
            now_latest_index = latest_index
            self.all_args.latest_faces = count_faces(self.all_args.photos_dir, now_latest_index)
            last_index = now_latest_index

            if self.all_args.B[t] > self.all_args.K_safe * self.all_args.B_max:
                self.alg_iterate(t)
            else:
                time.sleep(self.all_args.delta)

            self.all_args.save_data(res_filename, t)


class Active:
    def __init__(self, all_args):
        self.all_args = all_args

    def alg_iterate(self, t):
        global latest_index
        global last_index
        global start_index

        receive_photo_num = 0
        start_time = time.time()
        while time.time() - start_time < self.all_args.delta:
            if last_index == latest_index:  # 如果没有照片，循环等待
                time.sleep(1)
                continue

            faces_nums = count_faces(self.all_args.photos_dir, last_index + 1)
            rest_time = self.all_args.delta - (time.time() - start_time)
            need_processed_faces = faces_nums
            real_pm = decide_execute(rest_time, 1.8, 1, need_processed_faces, self.all_args.photos_dir)

            start_index += real_pm
            last_index += real_pm
            receive_photo_num += real_pm
        self.all_args.C[t] = receive_photo_num
        self.all_args.update_args(t, mu_t=1.8, pm_t=receive_photo_num, alg_name="active")

    def run(self, res_filename):
        with open(res_filename, "a") as f:
            f.write("\n" + str(datetime.datetime.now()) + "\n")
            f.write("t, X, C, PM, MU, B, E, L, lose_photo\n")

        for t in range(self.all_args.T):
            if self.all_args.B[t] > self.all_args.K_safe * self.all_args.B_max:
                self.alg_iterate(t)
            else:
                time.sleep(self.all_args.delta)
            self.all_args.save_data(res_filename, t)


class Lazy:
    def __init__(self, all_args):
        self.all_args = all_args

    def alg_iterate(self, t):
        global start_index

        start_time = time.time()

        mu_t, pm_t, E_u_t = 0, 0, 0
        if self.all_args.L[t] == self.all_args.L_max:
            self.all_args.X[t] = 1
            mu_t, pm_t, E_u_t = self.all_args.lazy_solve_xt(t)
        else:
            self.all_args.X[t] = 0

        # 执行决策，并更新参数
        during_time = int(time.time() - start_time)
        if self.all_args.X[t] == 1:
            need_processed_faces = list(self.all_args.L_Queue.queue)[:pm_t]
            real_pm = decide_execute(self.all_args.delta - during_time, mu_t, pm_t, need_processed_faces,
                                     self.all_args.photos_dir)
            start_index += real_pm  # 更新照片队列起始index

            self.all_args.update_args(t, mu_t=mu_t, pm_t=real_pm)
            # 如果执行决策后还剩余时间，就休眠
            during_time = time.time() - start_time
            sleep_time = self.all_args.delta - during_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"决策执行超时{sleep_time}秒")
        else:
            self.all_args.update_args(t)
            sleep_time = self.all_args.delta - during_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"决策执行超时{sleep_time}秒")

    def run(self, res_filename):
        global last_index
        global latest_index

        with open(res_filename, "a") as f:
            f.write("\n" + str(datetime.datetime.now()) + "\n")
            f.write("t, X, C, PM, MU, B, E, L, lose_photo\n")

        for t in range(self.all_args.T):
            now_latest_index = latest_index
            self.all_args.C[t] = now_latest_index - last_index
            self.all_args.latest_faces = count_faces(self.all_args.photos_dir, now_latest_index)
            last_index = now_latest_index

            if self.all_args.B[t] > self.all_args.K_safe * self.all_args.B_max:
                self.alg_iterate(t)
            else:
                time.sleep(self.all_args.delta)
            self.all_args.save_data(res_filename, t)


# 清点人脸
def count_faces(file_dir, count_latest_index):
    global last_index

    if not os.path.exists(file_dir + "/counted"):
        os.makedirs(file_dir + "/counted")

    faces_nums = []
    for i in range(last_index, count_latest_index):
        people_number = random.randint(1, 5)  # 此处为对比实验一致性，采用模拟数据
        example_filename = f"{file_dir}/example/{people_number}.jpg"
        image = cv2.imread(example_filename)

        origin_filename = f"{file_dir}/origin/{i}.jpg"
        os.remove(origin_filename)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_detect = cv2.CascadeClassifier('resource/haarcascade_frontalface_alt2.xml')
        faces = face_detect.detectMultiScale(gray, 1.01, 5, 0, (100, 100), (300, 300))
        faces_nums.append(len(faces))
        if len(faces) > 0:
            num = 1
            for x, y, w, h in faces:
                face_image = gray[y:y + h, x:x + w]
                count_filename = f"{file_dir}/counted/{i}-{num}.jpg"
                cv2.imwrite(count_filename, face_image)
                num += 1
    return faces_nums


def take_photo(file_dir):
    global latest_index
    camera = cv2.VideoCapture(0)

    if not os.path.exists(file_dir + "/origin"):
        os.makedirs(file_dir + "/origin")

    error_count = 0
    while True:
        ret, frame = camera.read()

        if not ret:
            print("无法读取摄像头数据！")
            error_count += 1
            if error_count == 10:
                break
            time.sleep(0.5)
            camera.release()
            cv2.destroyAllWindows()
            camera = cv2.VideoCapture(0)
            continue

        filename = f"{file_dir}/origin/{latest_index}.jpg"
        cv2.imwrite(filename, frame)

        latest_index += 1
        time.sleep(random.randint(6, 12))
    camera.release()
    cv2.destroyAllWindows()


# 启动人脸识别进程
def decide_execute(execute_time, mu_t, pm_t, faces, file_dir):
    global start_index

    with open("faces.txt", 'w') as file:
        for item in faces:
            file.write(str(item) + '\n')

    process = subprocess.Popen(
        ["python3", "decide_execute.py", str(execute_time), str(start_index), str(mu_t), str(pm_t), face_nums_filepath,
         file_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    print(stdout.decode())
    print(stderr.decode())
    real_pm = int(stdout.decode())
    return real_pm


start_index = 1  # 照片队列中第一张照片下标
last_index = 1  # 照片队列在上一时间片结束时的最后一张照片的下标
latest_index = 1  # 当前照片队列的最新照片下标


def alg_invoke(class_name, config_filename, photos_dir):
    with open(config_filename, 'r') as file:
        args = json.load(file)
    args["charge_filename"] = "../config/charge_data/mon_7/"
    args["res_dir"] = "../result/"
    args["photos_dir"] = photos_dir

    all_args = common_args.CommonArgs(args)
    if class_name in globals():
        algorithm_class = globals()[class_name]
        algorithm = algorithm_class(all_args)
        algorithm.run(all_args.res_dir + f"{class_name.lower()}.txt")
    else:
        print("算法名输入错误")


def main(current_alg_name):
    config_filename = "../config/common_config.json"
    photos_dir = "../photos"

    photo_thread = threading.Thread(target=take_photo, args=(photos_dir,))
    decide_thread = threading.Thread(target=alg_invoke, args=(current_alg_name, config_filename, photos_dir))

    photo_thread.start()
    decide_thread.start()

    photo_thread.join()
    decide_thread.join()


if __name__ == "__main__":
    alg_name = sys.argv[1]  # 算法名称 Ours Strict Active Lazy
    main(alg_name)
