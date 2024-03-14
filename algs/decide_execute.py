import sys
import cv2
import time
import os
import math

execute_time = float(sys.argv[1])  # 获取第一个参数值
start_index = int(sys.argv[2])
mu_t = float(sys.argv[3])  # 获取第二个参数值
pm_t = int(sys.argv[4])  # 获取第二个参数值
face_nums_filepath = sys.argv[5]
file_dir = sys.argv[6]
end_index = start_index + pm_t


def round_up(number, decimal_places):
    factor = 10 ** decimal_places
    rounded_number = math.ceil(number * factor) / factor
    return rounded_number


mu_t = round_up(mu_t, 1)  # 将mu_t向上进位，分辨率为0.1
command = "sudo cpufreq-set -f " + str(mu_t * 100000)
os.system('echo %s | sudo -S %s' % (123, command))
os.system(command)

# 加载已经训练好的人脸识别模型
model_path = 'resource/face_recognition_model.yml'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier('resource/haarcascade_frontalface_alt2.xml')

# 读取每张照片人脸数,从文本文件读取列表
face_nums = []
with open(face_nums_filepath, 'r') as file:
    for line in file:
        face_nums.append(int(line.strip()))

start_time = time.time()
photo_cou = 0
while time.time() - start_time < execute_time and start_index + photo_cou < end_index:
    for i in range(1, face_nums[photo_cou] + 1):
        photo_name = f"{file_dir}/counted/{start_index + photo_cou}-{i}.jpg"
        image = cv2.imread(photo_name, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(image, 1.01, 5, 0, (100, 100), (300, 300))
        os.remove(photo_name)  # 删掉文件
        label_id, confidence = recognizer.predict(image)
    photo_cou += 1

command = "sudo cpufreq-set -f 600000"
os.system('echo %s | sudo -S %s' % (123, command))
os.system(command)

print(photo_cou)  # 将结果通过print传递回去
