import numpy as np
import cv2
import os
from utils import linear_mapping, pre_process, random_warp

"""
This module implements the basic correlation filter based tracking algorithm -- MOSSE

Date: 2018-05-28

"""

class mosse:
    def __init__(self, args, img_path):
        # get arguments..
        self.args = args
        self.img_path = img_path
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()
    
    # start to do the object tracking...
    def start_tracking(self):
        # get the image of the first frame... (read as gray scale image...)
        # 读第一帧
        init_img = cv2.imread(self.frame_lists[0])
        #BGR2GRAY RGB转到灰度图
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        #np定义数据类型
        init_frame = init_frame.astype(np.float32)
        #自己选定GroundTruth，返回元组（矩形框最小的x值，矩形框最小的y值，矩形框的宽，矩形框的高）
        init_gt = cv2.selectROI('demo', init_img, False, False)
        init_gt = np.array(init_gt).astype(np.int64)
        # start to draw the gaussian response...
        response_map = self._get_gauss_response(init_frame, init_gt)
        ####################################################################################
        # start to create the training set ...
        # get the goal..
        #从响应图截取GT的位置
        g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        #从当前帧截取gt的位置
        fi = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        #做FFT
        G = np.fft.fft2(g)
        #########################################################################
        # start to do the pre-training...
        Ai, Bi = self._pre_training(fi, G)
        # 逐帧进行读取
        # start the tracking...
        for idx in range(len(self.frame_lists)):
            current_frame = cv2.imread(self.frame_lists[idx])
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)
            #如果是第一帧
            if idx == 0:
                #lr 学习率
                Ai = self.args.lr * Ai
                Bi = self.args.lr * Bi
                pos = init_gt.copy()
                clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64)
            else:
                Hi = Ai / Bi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                #预处理
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                #滤波窗口对于当前帧的响应图
                Gi = Hi * np.fft.fft2(fi)
                gi = linear_mapping(np.fft.ifft2(Gi))
                # find the max pos...
                max_value = np.max(gi)
                max_pos = np.where(gi == max_value)
                #对于原本GT的位移
                dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
                dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
                
                # update the position...
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy

                # trying to get the clipped position [xmin, ymin, xmax, ymax]
                #clip函数将pos限制在一个范围内 （大于第二个参数，小于第三个参数）
                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0]+pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1]+pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                # get the current fi..0
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                # online update...
                Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi
            
            # visualize the tracking process...
            #给当前帧画框
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
            cv2.imshow('demo', current_frame)
            cv2.waitKey(100)
            # if record... save the frames..
            if self.args.record:
                frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                if not os.path.exists(frame_path):
                    os.mkdir(frame_path)
                cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)


    # pre train the filter on the first frame...
    #G：高斯响应图截取GT位置做FFT
    #两个输出值Ai:相应图   Bi:原图像
    def _pre_training(self, init_frame, G):
        height, width = G.shape
        #将第一帧弄成GT大小
        #在主函数中形参init_frame传入的参数为帧根据GT位置截取大小
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        fi = pre_process(fi)
        #先搞一个共轭
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
        #Ai, Bi累加
        for _ in range(self.args.num_pretrain):
            # 看在预训练时图像有无旋转
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
        return Ai, Bi

    # get the ground-truth gaussian reponse...
    #返回关于GT中心点的响应函数，使得中心点的相应最大（原来都是平的，减到中心点的地方，使它最高）
    #返回的就是一个矩阵，每个数代表一个像素，GT所在位置数值最高
    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        # img 代表某帧图片
        height, width = img.shape
        # get the mesh grid...
        #生成一个图，每个网格点对应一个像素点
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # get the center of the object...
        #gt为groundtruth，返回的是元组包含四个元素，算出gt的中心的x,y坐标点
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)
        return response

    # it will extract the image list 
    def _get_img_lists(self, img_path):
        frame_list = []
        for frame in os.listdir(img_path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame)) 
        return frame_list
    
    # it will get the first ground truth of the video..
    def _get_init_ground_truth(self, img_path):
        gt_path = os.path.join(img_path, 'groundtruth.txt')
        with open(gt_path, 'r') as f:
            # just read the first frame...
            line = f.readline()
            gt_pos = line.split(',')
        return [float(element) for element in gt_pos]

