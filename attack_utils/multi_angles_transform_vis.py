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
dir_30_n_vis = "/workspace/spline_de_attack/result/tmp_dir_30_n_vis/"
dir_60_n_vis = "/workspace/spline_de_attack/result/tmp_dir_60_n_vis/" 
# dir_75_n_vis = "/workspace/spline_de_attack/result/tmp_dir_75_n_vis/"
dir_30_p_vis = "/workspace/spline_de_attack/result/tmp_dir_30_p_vis/"
dir_60_p_vis = "/workspace/spline_de_attack/result/tmp_dir_60_p_vis/"
# dir_75_p_vis = "/workspace/spline_de_attack/result/tmp_dir_75_p_vis/"
dir_dis_vis = "/workspace/spline_de_attack/result/tmp_dir_dis_vis/"

def img_process(img_path):
    img_sample = Image.open(img_path)
    img_input = trans(img_sample) # to tensor
    img_ori = torch.stack([img_input]) # N C H W
    return img_ori


def save_img(step_number, mask_dir, visible_img):
    mask = cv2.imread(mask_dir + '{}.png'.format(step_number), cv2.IMREAD_GRAYSCALE)    
    mask = np.array(mask) / 255
    mask = mask.astype(np.int8)
    mask = mask^(mask&1==mask)
    mask_ori = np.zeros((512,640))
    mask_ori[200:400,200:380] = mask[200:400,200:380]
    x_adv = visible_img * ( 1 - mask_ori ) + mask_ori * 1
    adv_final = x_adv[0].cpu().detach().numpy()
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    adv_sample = Image.fromarray(adv_x_255)
    save_path = mask_dir + '/{}.png'.format(step_number)
    adv_sample.save(save_path, quality=99)


def multi_angles_transform_vis(step_number):

    mask_image = cv2.imread(mask_dir +'/{}.png'.format(step_number))
    img_base_vis = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_visible/base.jpg")
    img_30_n_vis = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_visible/-30.jpg")
    # img_60_n_vis = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_visible/-60.jpg")
    # img_75_n_vis = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_visible/-75.jpg")
    img_30_p_vis = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_visible/+30.jpg")
    # img_60_p_vis = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_visible/+60.jpg")
    # img_75_p_vis = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_visible/+75.jpg")
    # img_dis_vis = cv2.imread("/workspace/spline_de_attack/physical_dataset/attack_visible/dis.jpg")
    
    imgRGB_base_vis = cv2.cvtColor(img_base_vis, cv2.COLOR_BGR2RGB)
    imgRGB_30_n_vis = cv2.cvtColor(img_30_n_vis, cv2.COLOR_BGR2RGB)
    # imgRGB_60_n_vis = cv2.cvtColor(img_60_n_vis, cv2.COLOR_BGR2RGB)
    # imgRGB_75_n_vis = cv2.cvtColor(img_75_n_vis, cv2.COLOR_BGR2RGB)
    imgRGB_30_p_vis = cv2.cvtColor(img_30_p_vis, cv2.COLOR_BGR2RGB)
    # imgRGB_60_p_vis = cv2.cvtColor(img_60_p_vis, cv2.COLOR_BGR2RGB)
    # imgRGB_75_p_vis = cv2.cvtColor(img_75_p_vis, cv2.COLOR_BGR2RGB)
    # imgRGB_dis_vis = cv2.cvtColor(img_dis_vis, cv2.COLOR_BGR2RGB)
    
    results_base_vis = pose.process(imgRGB_base_vis)
    results_30_n_vis = pose.process(imgRGB_30_n_vis)
    # results_60_n_vis = pose.process(imgRGB_60_n_vis)
    # results_75_n_vis = pose.process(imgRGB_75_n_vis)
    results_30_p_vis = pose.process(imgRGB_30_p_vis)
    # results_60_p_vis = pose.process(imgRGB_60_p_vis)
    # results_75_p_vis = pose.process(imgRGB_75_p_vis)
    # results_dis_vis = pose.process(imgRGB_dis_vis)
    
    # print(results.pose_landmarks)
    img_base_vis_list = []
    img_30_n_vis_list = []
    # img_60_n_vis_list = []
    # img_75_n_vis_list = []
    img_30_p_vis_list = []
    # img_60_p_vis_list = []
    # img_75_p_vis_list = []
    # img_dis_vis_list = []

    if results_base_vis.pose_landmarks:
        for id, lm in enumerate(results_base_vis.pose_landmarks.landmark):
            h, w, c = img_base_vis.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            img_base_vis_list.append([cx,cy])

    if results_30_n_vis.pose_landmarks:
        for id, lm in enumerate(results_30_n_vis.pose_landmarks.landmark):
            h, w, c = img_30_n_vis.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            img_30_n_vis_list.append([cx,cy])

    # if results_60_n_vis.pose_landmarks:
    #     for id, lm in enumerate(results_60_n_vis.pose_landmarks.landmark):
    #         h, w, c = img_60_n_vis.shape
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         img_60_n_vis_list.append([cx,cy])

    # if results_75_n_vis.pose_landmarks:
    #     for id, lm in enumerate(results_75_n_vis.pose_landmarks.landmark):
    #         h, w, c = img_75_n_vis.shape
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         img_75_n_vis_list.append([cx,cy])

    if results_30_p_vis.pose_landmarks:
        for id, lm in enumerate(results_30_p_vis.pose_landmarks.landmark):
            h, w, c = img_30_p_vis.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            img_30_p_vis_list.append([cx,cy])

    # if results_60_p_vis.pose_landmarks:
    #     for id, lm in enumerate(results_60_p_vis.pose_landmarks.landmark):
    #         h, w, c = img_60_p_vis.shape
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         img_60_p_vis_list.append([cx,cy])

    # if results_75_p_vis.pose_landmarks:
    #     for id, lm in enumerate(results_75_p_vis.pose_landmarks.landmark):
    #         h, w, c = img_75_p_vis.shape
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         img_75_p_vis_list.append([cx,cy])

    # if results_dis_vis.pose_landmarks:
    #     for id, lm in enumerate(results_dis_vis.pose_landmarks.landmark):
    #         h, w, c = img_dis_vis.shape
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         img_dis_vis_list.append([cx,cy])

    # 生成透视矩阵

    src_points = np.array(img_base_vis_list, dtype=np.float32)
    den_30_n_vis_points = np.array(img_30_n_vis_list, dtype=np.float32)
    # den_60_n_vis_points = np.array(img_60_n_vis_list, dtype=np.float32)
    # den_75_n_vis_points = np.array(img_75_n_vis_list, dtype=np.float32)
    den_30_p_vis_points = np.array(img_30_p_vis_list, dtype=np.float32)
    # den_60_p_vis_points = np.array(img_60_p_vis_list, dtype=np.float32)
    # den_75_p_vis_points = np.array(img_75_p_vis_list, dtype=np.float32)
    # den_dis_vis_points = np.array(img_dis_vis_list, dtype=np.float32)

    # 进行透视变换
    # 注意透视变换第三个参数为变换后图片大小，格式为（高度，宽度）
    # print(src_points,den_30_n_vis_points)
    h_30_n_vis, _ = cv2.findHomography(src_points, den_30_n_vis_points, cv2.RANSAC, 5.0)
    # h_60_n_vis, _ = cv2.findHomography(src_points, den_60_n_vis_points, cv2.RANSAC, 5.0)
    # h_75_n_vis, _ = cv2.findHomography(src_points, den_75_n_vis_points, cv2.RANSAC, 5.0)
    h_30_p_vis, _ = cv2.findHomography(src_points, den_30_p_vis_points, cv2.RANSAC, 5.0)
    # h_60_p_vis, _ = cv2.findHomography(src_points, den_60_p_vis_points, cv2.RANSAC, 5.0)
    # h_75_p_vis, _ = cv2.findHomography(src_points, den_75_p_vis_points, cv2.RANSAC, 5.0)
    # h_dis_vis, _ = cv2.findHomography(src_points, den_dis_vis_points, cv2.RANSAC, 5.0)

    # 进行透视变换
    warped_image_30_n_vis = cv2.warpPerspective(mask_image, h_30_n_vis, (640, 512))
    # warped_image_60_n_vis = cv2.warpPerspective(mask_image, h_60_n_vis, (640, 512))
    # warped_image_75_n_vis = cv2.warpPerspective(mask_image, h_75_n_vis, (640, 512))
    warped_image_30_p_vis = cv2.warpPerspective(mask_image, h_30_p_vis, (640, 512))
    # warped_image_60_p_vis = cv2.warpPerspective(mask_image, h_60_p_vis, (640, 512))
    # warped_image_75_p_vis = cv2.warpPerspective(mask_image, h_75_p_vis, (640, 512))
    # warped_image_dis_vis = cv2.warpPerspective(mask_image, h_dis_vis, (640, 512))

    cv2.imwrite(dir_30_n_vis + '{}.png'.format(step_number), warped_image_30_n_vis)
    cv2.imwrite('./{}_n.png'.format(step_number), warped_image_30_n_vis)
    # cv2.imwrite(dir_60_n_vis + '{}.png'.format(step_number), warped_image_60_n_vis)
    # cv2.imwrite(dir_75_n_vis + '{}.png'.format(step_number), warped_image_75_n_vis)
    cv2.imwrite(dir_30_p_vis + '{}.png'.format(step_number), warped_image_30_p_vis)
    cv2.imwrite('./{}_p.png'.format(step_number), warped_image_30_p_vis)
    # cv2.imwrite(dir_60_p_vis + '{}.png'.format(step_number), warped_image_60_p_vis)
    # cv2.imwrite(dir_75_p_vis + '{}.png'.format(step_number), warped_image_75_p_vis)
    # cv2.imwrite(dir_dis_vis + '{}.png'.format(step_number), warped_image_dis_vis)

    visible_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_visible/-30.jpg")
    save_img(step_number, dir_30_n_vis, visible_img)
    # visible_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_visible/-60.jpg")
    # save_img(step_number, dir_60_n_vis, visible_img)
    # visible_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_visible/-75.jpg")
    # save_img(step_number, dir_75_n_vis, visible_img)
    visible_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_visible/+30.jpg")
    save_img(step_number, dir_30_p_vis, visible_img)
    # visible_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_visible/+60.jpg")
    # save_img(step_number, dir_60_p_vis, visible_img)
    # visible_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_visible/+75.jpg")
    # save_img(step_number, dir_75_p_vis, visible_img)    
    # visible_img = img_process("/workspace/spline_de_attack/physical_dataset/attack_visible/dis.jpg")
    # save_img(step_number, dir_dis_vis, visible_img)




