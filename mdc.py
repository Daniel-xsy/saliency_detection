import math
import itertools
import cv2 as cv
import numpy as np
import utils

from datasets import ImageDataset

class MDCSaliency():
    def __init__(self, img):

        self.alpha = 0.8
        self.beta = 0.3
        self.theta = 0.

        self.original_map, self.aug_map = self._calculate_saliency(img)

    def _calculate_saliency(self, img):
        sum,sqsum = cv.integral2(img)
        area_sum = lambda sum,x1,x2,y1,y2 : (sum[y2,x2,:] - sum[y1,x2,:] - sum[y2,x1,:] + sum[y1,x1,:])

        h = img.shape[0]
        w = img.shape[1]
        S = np.zeros((h,w))
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                TL = np.sum(area_sum(sqsum,0,x+1,0,y+1)) - 2*np.sum(area_sum(sum,0,x+1,0,y+1)*img[y,x,:]) + (x+1)*(y+1)*np.sum(np.power(img[y,x,:],2,dtype=np.uint32))
                TR = np.sum(area_sum(sqsum,x,-1,0,y+1)) - 2*np.sum(area_sum(sum,x,-1,0,y+1)*img[y,x,:]) + (w-x)*(y+1)*np.sum(np.power(img[y,x,:],2,dtype=np.uint32))
                BL = np.sum(area_sum(sqsum,0,x+1,y,-1)) - 2*np.sum(area_sum(sum,0,x+1,y,-1)*img[y,x,:]) + (x+1)*(h-y)*np.sum(np.power(img[y,x,:],2,dtype=np.uint32))
                BR = np.sum(area_sum(sqsum,x,-1,y,-1)) - 2*np.sum(area_sum(sum,x,-1,y,-1)*img[y,x,:]) + (w-x)*(h-y)*np.sum(np.power(img[y,x,:],2,dtype=np.uint32))
                S[y,x] = np.sqrt(np.min([TL,TR,BL,BR]))
        
        S = S/np.max(S)*255
        gray  = S.astype(np.uint8)

        T,_ = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        marker = np.zeros((img.shape[0],img.shape[1]),dtype=np.int32)
        marker = np.where(gray>(1+self.theta)*T,1,marker)
        marker = np.where(gray<(1-self.theta)*T,2,marker)

        marker = cv.watershed(img,marker)

        S_enhance = np.where(marker==1,1-self.alpha*(1-S),S)
        S_enhance = np.where(marker==2,self.alpha*S,S)
        return S, S_enhance