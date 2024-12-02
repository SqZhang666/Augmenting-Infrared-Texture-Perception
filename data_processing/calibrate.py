#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/11 16:23
# @File    : calibrate.py
# @Description : 两个相机图像校准
import cv2
import numpy as np


def calibrate():
    camera_matrix = np.array([
              [ 403.142154163, 0, 320 ],
              [ 0, 403.142154163, 256 ],
              [ 0, 0, 1 ]
            ])
    #dist_coeffs = np.array([1.0, 2.0, -0.0, -0.0, 0.1, 2.3, 2.5, 0.6, 0, 0, 0, 0, 0, 0 ])
    dist_coeffs = np.array([-0.10386666,0,0,0])
    distorted_img = cv2.imread(r'001.jpg')
    undistorted_img = cv2.undistort(distorted_img, camera_matrix, dist_coeffs, None, camera_matrix)

    cv2.imshow('distorted_img', distorted_img)
    cv2.imshow('undistorted_img', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    calibrate()