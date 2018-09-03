#
#
# import numpy as np
# import cv2
#
# input_image_full_path = '../dataset/chicken.png'
# input_image_full_path = '../dataset/train/masks/0a7e067255.png'
# image = cv2.imread(input_image_full_path, cv2.IMREAD_GRAYSCALE)
#
# print image
# print 1-image
#
# h_flip = cv2.flip(image, 1)
# v_flip = cv2.flip(image, 0)
# rot90 = np.rot90(image)
# rot180 = np.rot90(np.rot90(image))
# rot270 = np.rot90(np.rot90(np.rot90(image)))
#
# print image.shape
#
#
# # cv2.imshow('raw.jpg', image)
# # cv2.imshow('h_flip.jpg', h_flip)
# # cv2.imshow('v_flip.jpg', v_flip)
# # cv2.imshow("rot90.jpg", rot90)
# # cv2.imshow("rot180.jpg", rot180)
# # cv2.imshow("rot270.jpg", rot270)
# #
# # cv2.waitKey(0)
#


file = 'log.txt'
wf = open(file, 'wb')
wf.write('id,rle_mask\n')
wf.write('id,rle_mask\n')
wf.close()



