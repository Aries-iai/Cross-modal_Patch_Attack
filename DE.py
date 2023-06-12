import os
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import math
from attack_utils.spline import spline_multi_mask, get_multi_mask
import cv2
import torch
import torch.nn as nn 
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from yolov3.detect_infrared import detect_infrared
from yolov3.detect_visible import detect_visible

tmp_dir_inf = '/workspace/cross_modal_patch_attack/result/tmp_dir_infrared'
tmp_dir_vis = '/workspace/cross_modal_patch_attack/result/tmp_dir_visible'
mask_dir = '/workspace/cross_modal_patch_attack/result/mask'
final_dir = '/workspace/cross_modal_patch_attack/result/final'

trans = transforms.Compose([
                transforms.ToTensor(),
            ])
content_inf = 0
content_vis = 1

def ifcross(p1, p2, p, px, py):
    d1 = (p1[0]-px)*(p[1]-py) - (p1[1]-py)*(p[0]-px)
    d2 = (p2[0]-px)*(p[1]-py) - (p2[1]-py)*(p[0]-px)
    if d1 * d2 < 0:
        return False
    else:
        return True

def compute_dis(p, px, py):
    dis = pow(pow(p[0]-px, 2) + pow(p[1]-py, 2), 0.5)
    return dis

def GrieFunc(vardim, x, infrared_ori, visible_ori, threat_infrared_model, threat_visible_model, prob_ori_infrared, prob_ori_visible, img_name, step_number, h, w):
    state = []
    p1 = []
    p2 = []
    length = int(len(x)/2)
    for i in range(length):
        if i % 2 == 0:
            p1.append([x[i], x[i+1]])
    for i in range(length, 2*length):
        if i % 2 == 0:
            p2.append([x[i], x[i+1]])
    state.append(p1)
    state.append(p2)
    mask = spline_multi_mask(state, h, w)
    len_x = len(mask)
    len_y = len(mask[0])

    # obtain the mask
    mask = trans(mask)
    mask = mask[0].cpu().detach().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask_path = mask_dir + '/{}.png'.format(step_number)
    mask.save(mask_path, quality = 99)
    mask = cv2.imread(mask_path)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)
    cv2.imwrite(mask_dir+'/{}.png'.format(step_number), mask)
    
    # cast mask upon infrared images
    fig = mask_dir +'/{}.png'.format(step_number)
    mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask) / 255
    mask = mask.astype(np.int8)
    mask = mask^(mask&1==mask)
    x_adv = infrared_ori * ( 1 - mask ) + mask * content_inf
    adv_final = x_adv[0].cpu().detach().numpy()
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    adv_sample = Image.fromarray(adv_x_255)
    save_path = tmp_dir_inf + '/{}.png'.format(step_number)
    adv_sample.save(save_path, quality=99)

    # cast mask upon visible images
    fig = mask_dir +'/{}.png'.format(step_number)
    mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask) / 255
    mask = mask.astype(np.int8)
    mask = mask^(mask&1==mask)
    x_adv = visible_ori * ( 1 - mask ) + mask * content_vis
    adv_final = x_adv[0].cpu().detach().numpy()
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    adv_sample = Image.fromarray(adv_x_255)
    save_path = tmp_dir_vis + '/{}.png'.format(step_number)
    adv_sample.save(save_path, quality=99)
    
    # r_attack
    with open(tmp_dir_inf + '/{}.png'.format(step_number), 'rb') as fig:
        sample = Image.open(fig)
        infrared_input = trans(sample)
        infrared_ori = torch.stack([infrared_input]) # N C H W
        infrared_det = F.interpolate(infrared_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
        _, prob_infrared = detect_infrared(threat_infrared_model, infrared_det)  

    with open(tmp_dir_vis + '/{}.png'.format(step_number), 'rb') as fig:
        sample = Image.open(fig)
        visible_input = trans(sample)
        visible_ori = torch.stack([visible_input]) # N C H W
        visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小  
        _, prob_visible = detect_visible(threat_visible_model, visible_det) 

    r_inf = math.exp(2*((prob_ori_infrared - prob_infrared) / (prob_ori_infrared - 0.7)))
    r_vis = math.exp(2*((prob_ori_visible - prob_visible) / (prob_ori_visible - 0.7)))
    r_attack =  min(r_inf, r_vis)

    dis_inf = np.float((prob_ori_infrared - prob_infrared) / (prob_ori_infrared - 0.7))
    dis_vis = np.float((prob_ori_visible - prob_visible) / (prob_ori_visible - 0.7))
    dis_to_success = min(dis_inf, dis_vis)



    return r_attack, dis_to_success, dis_inf, dis_vis


class DEIndividual:

    '''
    individual of differential evolution algorithm
    '''

    def __init__(self,  vardim, points):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.points = points
        self.fitness = 0.
        self.distance = 0.
        self.dis_inf = 0.
        self.dis_vis = 0.

    def generate(self):
        '''
        generate a random chromsome for differential evolution algorithm
        '''
        len = self.vardim
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.points[i] + np.random.randint(-3, 3)
        # print(self.chrom)
                        

    def calculateFitness(self, infrared_ori, visible_ori, threat_infrared_model, threat_visible_model, prob_ori_infrared, prob_ori_visible, img_name, step_number, h, w):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness, self.distance, self.dis_inf, self.dis_vis = GrieFunc(
            self.vardim, self.chrom, infrared_ori, visible_ori, threat_infrared_model, threat_visible_model, prob_ori_infrared, prob_ori_visible, img_name, step_number, h, w)

class DifferentialEvolutionAlgorithm:

    '''
    The class for differential evolution algorithm
    '''

    def __init__(self, sizepop, vardim, points, eq_points, circle, region, infrared_ori, visible_ori, threat_infrared_model, threat_visible_model, prob_ori_infrared,\
         prob_ori_visible, img_name, MAXGEN, params, h, w):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        param: algorithm required parameters, it is a list which is consisting of [crossover rate CR, scaling factor F]
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.points = points
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params
        self.circle = circle
        self.eq_points = eq_points
        self.region = region
        self.infrared_ori = infrared_ori
        self.visible_ori = visible_ori
        self.threat_infrared_model = threat_infrared_model
        self.threat_visible_model = threat_visible_model
        self.prob_ori_infrared = prob_ori_infrared
        self.prob_ori_visible = prob_ori_visible
        self.img_name = img_name
        self.step_number = 0
        self.h = h
        self.w = w

    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = DEIndividual(self.vardim, self.points)
            ind.generate()
            self.population.append(ind)

    def evaluate(self, x):
        '''
        evaluation of the population fitnesses
        '''
        x.calculateFitness(self.infrared_ori, self.visible_ori, self.threat_infrared_model, self.threat_visible_model, self.prob_ori_infrared,\
             self.prob_ori_visible, self.img_name, self.step_number, self.h, self.w)

    def solve(self):
        '''
        evolution process of differential evolution algorithm
        '''
        self.step_number = 0
        self.t = 0
        self.initialize()
        for i in range(0, self.sizepop):
            self.evaluate(self.population[i])
            self.step_number += 1
            self.fitness[i] = self.population[i].fitness
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = self.best.fitness
        self.trace[self.t, 1] = self.avefitness
        print("Generation %d: optimal function value is: %f; distance to success is %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.best.distance, self.trace[self.t, 1]))
        print("dis_inf:{}, dis_vis:{}".format(self.best.dis_inf, self.best.dis_vis))

        x = self.best.chrom
        state = []
        p1 = []
        p2 = []
        length = int(len(x)/2)
        for i in range(length):
            if i % 2 == 0:
                p1.append([x[i], x[i+1]])
        for i in range(length, 2*length):
            if i % 2 == 0:
                p2.append([x[i], x[i+1]])
        state.append(p1)
        state.append(p2)
        mask = spline_multi_mask(state, self.h, self.w)

        # obtain the mask
        mask = trans(mask)
        mask = mask[0].cpu().detach().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask)
        mask_path = optimical_dir + '/{}_{}.png'.format(self.img_name, self.t)
        mask.save(mask_path, quality = 99)
        mask = cv2.imread(mask_path)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)
        
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            for i in range(0, self.sizepop):
                vi = self.mutationOperation(i)                                                                                                                             
                ui = self.crossoverOperation(i, vi)
                xi_next = self.selectionOperation(i, ui)
                self.population[i] = xi_next
            for i in range(0, self.sizepop):
                self.fitness[i] = self.population[i].fitness
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            print("Generation %d: optimal function value is: %f; distance to success is %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.best.distance, self.trace[self.t, 1]))
            print("dis_inf:{}, dis_vis:{}".format(self.best.dis_inf, self.best.dis_vis))                
            if self.best.fitness >= math.exp(2) or self.t == self.MAXGEN - 1:
                x = self.best.chrom
                state = []
                p1 = []
                p2 = []
                length = int(len(x)/2)
                for i in range(length):
                    if i % 2 == 0:
                        p1.append([x[i], x[i+1]])
                for i in range(length, 2*length):
                    if i % 2 == 0:
                        p2.append([x[i], x[i+1]])
                state.append(p1)
                state.append(p2)
                mask = spline_multi_mask(state, self.h, self.w)

                # obtain the mask
                mask = trans(mask)
                mask = mask[0].cpu().detach().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = Image.fromarray(mask)
                mask_path = mask_dir + '/{}.png'.format(self.img_name)
                mask.save(mask_path, quality = 99)
                mask = cv2.imread(mask_path)
                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)
                cv2.imwrite(mask_dir+'/{}.png'.format(self.img_name), mask)
                
                # cast mask upon infrared images
                fig = mask_dir +'/{}.png'.format(self.img_name)
                mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
                mask = np.array(mask) / 255
                mask = mask.astype(np.int8)
                mask = mask^(mask&1==mask)
                x_adv = self.infrared_ori * ( 1 - mask ) + mask * content_inf
                adv_final = x_adv[0].cpu().detach().numpy()
                adv_final = (adv_final * 255).astype(np.uint8)
                adv_x_255 = np.transpose(adv_final, (1, 2, 0))
                adv_sample = Image.fromarray(adv_x_255)
                save_path = final_dir + '/infrared_{}_{}.png'.format(self.img_name, self.best.fitness)
                adv_sample.save(save_path, quality=99)

                # cast mask upon visible images
                fig = mask_dir +'/{}.png'.format(self.img_name)
                mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
                mask = np.array(mask) / 255
                mask = mask.astype(np.int8)
                mask = mask^(mask&1==mask)
                x_adv = self.visible_ori * ( 1 - mask ) + mask * content_vis
                adv_final = x_adv[0].cpu().detach().numpy()
                adv_final = (adv_final * 255).astype(np.uint8)
                adv_x_255 = np.transpose(adv_final, (1, 2, 0))
                adv_sample = Image.fromarray(adv_x_255)
                save_path = final_dir + '/visible_{}_{}.png'.format(self.img_name, self.best.fitness)
                adv_sample.save(save_path, quality=99)

                with open(final_dir + '/infrared_{}_{}.png'.format(self.img_name, self.best.fitness), 'rb') as fig:
                    sample = Image.open(fig)
                    infrared_input = trans(sample)
                    infrared_ori = torch.stack([infrared_input]) # N C H W
                    infrared_det = F.interpolate(infrared_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
                    _, prob_infrared = detect_infrared(self.threat_infrared_model, infrared_det)  

                with open(final_dir + '/visible_{}_{}.png'.format(self.img_name, self.best.fitness), 'rb') as fig:
                    sample = Image.open(fig)
                    visible_input = trans(sample)
                    visible_ori = torch.stack([visible_input]) # N C H W
                    visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小  
                    _, prob_visible = detect_visible(self.threat_visible_model, visible_det) 
                os.rename(final_dir + '/infrared_{}_{}.png'.format(self.img_name, self.best.fitness), final_dir + '/infrared_{}_{}_{}.png'.format(self.img_name, self.t, prob_infrared))
                os.rename(final_dir + '/visible_{}_{}.png'.format(self.img_name, self.best.fitness), final_dir + '/visible_{}_{}_{}.png'.format(self.img_name, self.t, prob_visible))                
                break

        print("Optimal function value is: %f; " %\
              self.trace[self.t, 0])
        print ('Optimal solution is:')
        print (self.best.chrom)
        # self.printResult()

    def selectionOperation(self, i, ui):
        '''
        selection operation for differential evolution algorithm
        '''
        xi_next = copy.deepcopy(self.population[i])
        xi_next.chrom = ui
        self.evaluate(xi_next)
        self.step_number += 1
        if xi_next.fitness > self.population[i].fitness:
            # print("change")
            return xi_next
        else:
            # print("no change")
            return self.population[i]

    def crossoverOperation(self, i, vi):
        '''
        crossover operation for differential evolution algorithm
        '''
        px_1, py_1, px_2, py_2 = self.circle[0], self.circle[1], self.circle[2], self.circle[3]
        y_low, y_high, x_left, x_right = self.region[0], self.region[1], self.region[2], self.region[3]
        limit_line = (px_1 + px_2) / 2
        k = np.random.randint(0, self.vardim - 1)
        ui = np.zeros(self.vardim)
        for j in range(0, int(self.vardim/2)):
            if j % 2 == 0:
                dis = compute_dis([vi[j], vi[j + 1]], px_1, py_1)
                pick = random.random()
                if (pick < self.params[0] or j == k) and (ifcross(self.eq_points[0][j // 2], self.eq_points[0][j // 2 + 1], [vi[j], vi[j+1]], px_1, py_1) ==False \
                and dis > 8 and vi[j] >  y_low and vi[j] < limit_line and vi[j+1] > x_left and vi[j+1] < x_right):
                    ui[j] = vi[j]
                    ui[j + 1] = vi[j + 1]
                else:
                    ui[j] = self.population[i].chrom[j]
                    ui[j + 1] = self.population[i].chrom[j+1]
        for j in range(int(self.vardim/2) ,int(self.vardim)):
            if j % 2 == 0:
                dis = compute_dis([vi[j], vi[j + 1]], px_1, py_1)
                pick = random.random()
                if (pick < self.params[0] or j == k) and (ifcross(self.eq_points[1][(j - self.vardim // 2) // 2], self.eq_points[1][(j - self.vardim // 2) // 2 + 1], [vi[j], vi[j+1]], px_2, py_2) ==False \
                and dis > 8 and vi[j] >  limit_line and vi[j] < y_high and vi[j+1] > x_left and vi[j+1] < x_right):
                    ui[j] = vi[j]
                    ui[j + 1] = vi[j + 1]
                else:
                    ui[j] = self.population[i].chrom[j]
                    ui[j + 1] = self.population[i].chrom[j+1]
                    
        return ui

    def mutationOperation(self, i):
        '''
        mutation operation for differential evolution algorithm
        '''
        a = np.random.randint(0, self.sizepop - 1)
        while a == i:
            a = np.random.randint(0, self.sizepop - 1)
        b = np.random.randint(0, self.sizepop - 1)
        while b == i or b == a:
            b = np.random.randint(0, self.sizepop - 1)
        c = np.random.randint(0, self.sizepop - 1)
        while c == i or c == b or c == a:
            c = np.random.randint(0, self.sizepop - 1)
        vi = self.population[c].chrom + self.params[1] * \
            (self.population[a].chrom - self.population[b].chrom)

        return vi

    def printResult(self):
        '''
        plot the result of the differential evolution algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Differential Evolution Algorithm for function optimization")
        plt.legend()
        plt.show()
