import numpy as np
import cv2
import os
import argparse



def gen_label_for_npy(args):
    H, W = 48, 40
    txt_path = os.path.join(args.root, '/list/train_gt.txt')
    count = 0
    with open(txt_path) as f:
        for line in f:
            count+=1
            print(count)
            label_path = line.split()[1]
            seg_label_path = os.path.join(args.root, label_path)
            seg_label_img = cv2.imread(seg_label_path)
            seg_label_img = seg_label_img[:, :, 0]
            seg_img = np.zeros((3, H, W))
            h_samples = [115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235, 245, 255, 265,
                    275, 285, 295, 305, 315, 325, 335, 345, 355, 365, 375, 385, 395, 405, 415, 425, 435,
                    445, 455, 465, 475, 485, 495, 505, 515, 525, 535, 545, 555, 565, 575, 585]
            for i in range(48):
                ground = seg_label_img[h_samples[i],:]
                local1 = np.mean(np.where(ground == 1))
                local2 = np.mean(np.where(ground == 2))
                local3 = np.mean(np.where(ground == 3))
                local4 = np.mean(np.where(ground == 4))
                if local1>0:
                    seg_img[0, i, int(local1 / 41)] = 1
                    seg_img[1, i, int(local1 / 41)] = 1
                    seg_img[2, i, int(local1 / 41)] = (local1 % 41) / 41
                if local2>0:
                    seg_img[0, i, int(local2 / 41)] = 1
                    seg_img[1, i, int(local2 / 41)] = 2
                    seg_img[2, i, int(local2 / 41)] = (local2 % 41) / 41
                if local3>0:
                    seg_img[0, i, int(local3 / 41)] = 1
                    seg_img[1, i, int(local3 / 41)] = 3
                    seg_img[2, i, int(local3 / 41)] = (local3 % 41) / 41
                if local4>0:
                    seg_img[0, i, int(local4 / 41)] = 1
                    seg_img[1, i, int(local4 / 41)] = 4
                    seg_img[2, i, int(local4 / 41)] = (local4 % 41) / 41

            seg_path = seg_label_path[:-3] + 'npy'
            np.save(seg_path, seg_img)







def generate_label(args):
    print("generating test set...")
    gen_label_for_npy(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='D:',  help='The root of the culane dataset')
    args = parser.parse_args()
    generate_label(args)
