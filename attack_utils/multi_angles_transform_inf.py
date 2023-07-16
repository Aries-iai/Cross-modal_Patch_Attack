import cv2
import mediapipe as mp
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

trans = transforms.Compose([
                transforms.ToTensor(),
            ])
mpPose = mp.solutions.pose
pose = mpPose.Pose()

mask_dir = '/workspace/spline_de_attack/result/mask'
dir_30_n_inf = "/workspace/spline_de_attack/result/tmp_dir_30_n_inf/"
dir_60_n_inf = "/workspace/spline_de_attack/result/tmp_dir_60_n_inf/" 
# dir_75_n_inf = "/workspace/spline_de_attack/result/tmp_dir_75_n_inf/"
dir_30_p_inf = "/workspace/spline_de_attack/result/tmp_dir_30_p_inf/"
dir_60_p_inf = "/workspace/spline_de_attack/result/tmp_dir_60_p_inf/"
# dir_75_p_inf = "/workspace/spline_de_attack/result/tmp_dir_75_p_inf/"
dir_dis_inf = "/workspace/spline_de_attack/result/tmp_dir_dis_inf/"

def img_process(img_path):
    img_sample = Image.open(img_path)
    img_input = trans(img_sample) # to tensor
    img_ori = torch.stack([img_input]) # N C H W
    return img_ori


def save_img(step_number, mask_dir, infrared_img):
    mask = cv2.imread(mask_dir + '{}.png'.format(step_number), cv2.IMREAD_GRAYSCALE)    
    mask = np.array(mask) / 255
    mask = mask.astype(np.int8)
    mask = mask^(mask&1==mask)
    mask_ori = np.zeros((512,640))
    mask_ori[200:400,250:380] = mask[200:400,250:380]
    x_adv = infrared_img * ( 1 - mask_ori ) + mask_ori * 0
    adv_final = x_adv[0].cpu().detach().numpy()
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    adv_sample = Image.fromarray(adv_x_255)
    save_path = mask_dir + '/{}.png'.format(step_number)
    adv_sample.save(save_path, quality=99)


def multi_angles_transform_inf(step_number):

    mask_image = cv2.imread(mask_dir +'/{}.png'.format(step_number))
    img_base_inf = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_infrared/base.jpg")
    img_30_n_inf = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_infrared/-30.jpg")
    # img_60_n_inf = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_infrared/-60.jpg")
    # img_75_n_inf = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_infrared/-75.jpg")
    img_30_p_inf = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_infrared/+30.jpg")
    # img_60_p_inf = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_infrared/+60.jpg")
    # img_75_p_inf = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_infrared/+75.jpg")
    # img_dis_inf = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_infrared/dis.jpg")
    
    imgRGB_base_inf = cv2.cvtColor(img_base_inf, cv2.COLOR_BGR2RGB)
    imgRGB_30_n_inf = cv2.cvtColor(img_30_n_inf, cv2.COLOR_BGR2RGB)
    # imgRGB_60_n_inf = cv2.cvtColor(img_60_n_inf, cv2.COLOR_BGR2RGB)
    # imgRGB_75_n_inf = cv2.cvtColor(img_75_n_inf, cv2.COLOR_BGR2RGB)
    imgRGB_30_p_inf = cv2.cvtColor(img_30_p_inf, cv2.COLOR_BGR2RGB)
    # imgRGB_60_p_inf = cv2.cvtColor(img_60_p_inf, cv2.COLOR_BGR2RGB)
    # imgRGB_75_p_inf = cv2.cvtColor(img_75_p_inf, cv2.COLOR_BGR2RGB)
    # imgRGB_dis_inf = cv2.cvtColor(img_dis_inf, cv2.COLOR_BGR2RGB)
    
    results_base_inf = pose.process(imgRGB_base_inf)
    results_30_n_inf = pose.process(imgRGB_30_n_inf)
    # results_60_n_inf = pose.process(imgRGB_60_n_inf)
    # results_75_n_inf = pose.process(imgRGB_75_n_inf)
    results_30_p_inf = pose.process(imgRGB_30_p_inf)
    # results_60_p_inf = pose.process(imgRGB_60_p_inf)
    # results_75_p_inf = pose.process(imgRGB_75_p_inf)
    # results_dis_inf = pose.process(imgRGB_dis_inf)
    
    # print(results.pose_landmarks)
    img_base_inf_list = []
    img_30_n_inf_list = []
    # img_60_n_inf_list = []
    # img_75_n_inf_list = []
    img_30_p_inf_list = []
    # img_60_p_inf_list = []
    # img_75_p_inf_list = []
    # img_dis_inf_list = []

    if results_base_inf.pose_landmarks:
        for id, lm in enumerate(results_base_inf.pose_landmarks.landmark):
            h, w, c = img_base_inf.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            img_base_inf_list.append([cx,cy])

    if results_30_n_inf.pose_landmarks:
        for id, lm in enumerate(results_30_n_inf.pose_landmarks.landmark):
            h, w, c = img_30_n_inf.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            img_30_n_inf_list.append([cx,cy])

    # if results_60_n_inf.pose_landmarks:
    #     for id, lm in enumerate(results_60_n_inf.pose_landmarks.landmark):
    #         h, w, c = img_60_n_inf.shape
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         img_60_n_inf_list.append([cx,cy])

    # if results_75_n_inf.pose_landmarks:
    #     for id, lm in enumerate(results_75_n_inf.pose_landmarks.landmark):
    #         h, w, c = img_75_n_inf.shape
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         img_75_n_inf_list.append([cx,cy])

    if results_30_p_inf.pose_landmarks:
        for id, lm in enumerate(results_30_p_inf.pose_landmarks.landmark):
            h, w, c = img_30_p_inf.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            img_30_p_inf_list.append([cx,cy])

    # if results_60_p_inf.pose_landmarks:
    #     for id, lm in enumerate(results_60_p_inf.pose_landmarks.landmark):
    #         h, w, c = img_60_p_inf.shape
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         img_60_p_inf_list.append([cx,cy])

    # if results_75_p_inf.pose_landmarks:
    #     for id, lm in enumerate(results_75_p_inf.pose_landmarks.landmark):
    #         h, w, c = img_75_p_inf.shape
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         img_75_p_inf_list.append([cx,cy])

    # if results_dis_inf.pose_landmarks:
    #     for id, lm in enumerate(results_dis_inf.pose_landmarks.landmark):
    #         h, w, c = img_dis_inf.shape
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         img_dis_inf_list.append([cx,cy])

    # 生成透视矩阵

    src_points = np.array(img_base_inf_list, dtype=np.float32)
    den_30_n_inf_points = np.array(img_30_n_inf_list, dtype=np.float32)
    # den_60_n_inf_points = np.array(img_60_n_inf_list, dtype=np.float32)
    # den_75_n_inf_points = np.array(img_75_n_inf_list, dtype=np.float32)
    den_30_p_inf_points = np.array(img_30_p_inf_list, dtype=np.float32)
    # den_60_p_inf_points = np.array(img_60_p_inf_list, dtype=np.float32)
    # den_75_p_inf_points = np.array(img_75_p_inf_list, dtype=np.float32)
    # den_dis_inf_points = np.array(img_dis_inf_list, dtype=np.float32)

    # 进行透视变换
    # 注意透视变换第三个参数为变换后图片大小，格式为（高度，宽度）
    # print(src_points,den_30_n_inf_points)
    h_30_n_inf, _ = cv2.findHomography(src_points, den_30_n_inf_points, cv2.RANSAC, 5.0)
    # h_60_n_inf, _ = cv2.findHomography(src_points, den_60_n_inf_points, cv2.RANSAC, 5.0)
    # h_75_n_inf, _ = cv2.findHomography(src_points, den_75_n_inf_points, cv2.RANSAC, 5.0)
    h_30_p_inf, _ = cv2.findHomography(src_points, den_30_p_inf_points, cv2.RANSAC, 5.0)
    # h_60_p_inf, _ = cv2.findHomography(src_points, den_60_p_inf_points, cv2.RANSAC, 5.0)
    # h_75_p_inf, _ = cv2.findHomography(src_points, den_75_p_inf_points, cv2.RANSAC, 5.0)
    # h_dis_inf, _ = cv2.findHomography(src_points, den_dis_inf_points, cv2.RANSAC, 5.0)

    # 进行透视变换
    warped_image_30_n_inf = cv2.warpPerspective(mask_image, h_30_n_inf, (640, 512))
    # warped_image_60_n_inf = cv2.warpPerspective(mask_image, h_60_n_inf, (640, 512))
    # warped_image_75_n_inf = cv2.warpPerspective(mask_image, h_75_n_inf, (640, 512))
    warped_image_30_p_inf = cv2.warpPerspective(mask_image, h_30_p_inf, (640, 512))
    # warped_image_60_p_inf = cv2.warpPerspective(mask_image, h_60_p_inf, (640, 512))
    # warped_image_75_p_inf = cv2.warpPerspective(mask_image, h_75_p_inf, (640, 512))
    # warped_image_dis_inf = cv2.warpPerspective(mask_image, h_dis_inf, (640, 512))

    cv2.imwrite(dir_30_n_inf + '{}.png'.format(step_number), warped_image_30_n_inf)
    # cv2.imwrite(dir_60_n_inf + '{}.png'.format(step_number), warped_image_60_n_inf)
    # cv2.imwrite(dir_75_n_inf + '{}.png'.format(step_number), warped_image_75_n_inf)
    cv2.imwrite(dir_30_p_inf + '{}.png'.format(step_number), warped_image_30_p_inf)
    # cv2.imwrite(dir_60_p_inf + '{}.png'.format(step_number), warped_image_60_p_inf)
    # cv2.imwrite(dir_75_p_inf + '{}.png'.format(step_number), warped_image_75_p_inf)
    # cv2.imwrite(dir_dis_inf + '{}.png'.format(step_number), warped_image_dis_inf)

    infrared_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_infrared/-30.jpg")
    save_img(step_number, dir_30_n_inf, infrared_img)
    # infrared_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_infrared/-60.jpg")
    # save_img(step_number, dir_60_n_inf, infrared_img)
    # infrared_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_infrared/-75.jpg")
    # save_img(step_number, dir_75_n_inf, infrared_img)
    infrared_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_infrared/+30.jpg")
    save_img(step_number, dir_30_p_inf, infrared_img)
    # infrared_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_infrared/+60.jpg")
    # save_img(step_number, dir_60_p_inf, infrared_img)
    # infrared_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_infrared/+75.jpg")
    # save_img(step_number, dir_75_p_inf, infrared_img)    
    # infrared_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_infrared/dis.jpg")
    # save_img(step_number, dir_dis_inf, infrared_img)




