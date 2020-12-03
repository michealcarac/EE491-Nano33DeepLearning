# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:44:15 2019

@author: ocasciotti
"""

import cv2, os
os.chdir("Dataset")
folder = 'fifty'
os.mkdir(folder)
os.chdir(folder)
cap1 = cv2.VideoCapture(1)

picture = 1

while True:
    ret1, frame1 = cap1.read()
    cv2.imshow('frame1' , frame1)
    
    key = cv2.waitKey(1) & 0xFF
    if ret1 and key == ord('p'):
        cv2.imwrite( folder + '_' + str(picture) + '.png', frame1)
        print('Pictures Taken: ' + str(picture))
        picture += 1
    elif key == ord('q'):
        cap1.release()
        cv2.destroyAllWindows()
        break
