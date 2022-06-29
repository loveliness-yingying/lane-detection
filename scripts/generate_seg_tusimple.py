import json
import numpy as np
import cv2
import os
import argparse

TRAIN_SET = ['label_data_0313.json', 'label_data_0601.json']
VAL_SET = ['label_data_0531.json']
TRAIN_VAL_SET = TRAIN_SET + VAL_SET
TEST_SET = ['test_tasks_0627.json']

def gen_label_for_json(args, image_set):
    H, W = 56, 32

    json_path = os.path.join(args.root,  "{}.json".format(image_set))
    with open(json_path) as f:
        for line in f:
            label = json.loads(line)
            # ---------- clean and sort lanes -------------
            lanes = []
            _lanes = []
            slope = [] # identify 0th, 1st, 2nd, 3rd, 4th, 5th lane through slope
            for i in range(len(label['lanes'])):
                l = [(x, y) for x, y in zip(label['lanes'][i], label['h_samples']) if x >= 0]
                if (len(l)>1):
                    _lanes.append(l)
                    slope.append(np.arctan2(l[-1][1]-l[0][1], l[0][0]-l[-1][0]) / np.pi * 180)
            _lanes = [_lanes[i] for i in np.argsort(slope)]
            slope = [slope[i] for i in np.argsort(slope)]

            idx = [None for i in range(6)]
            for i in range(len(slope)):
                if slope[i] <= 90:
                    idx[2] = i
                    idx[1] = i-1 if i > 0 else None
                    idx[0] = i-2 if i > 1 else None
                else:
                    idx[3] = i
                    idx[4] = i+1 if i+1 < len(slope) else None
                    idx[5] = i+2 if i+2 < len(slope) else None
                    break
            for i in range(6):
                lanes.append([] if idx[i] is None else _lanes[idx[i]])

            # ---------------------------------------------

            img_path = label['raw_file']
            seg_img = np.zeros((3, H, W))

            for i in range(len(lanes)):
                coords = lanes[i]
                if len(coords) < 4:
                    #list_str.append('0')
                    continue
                for j in range(len(coords)-1):
                    seg_img[0,int((coords[j][1]-160)/10), int(coords[j][0]/40)] = 1
                    seg_img[1,int((coords[j][1] - 160) / 10), int(coords[j][0] / 40)] = i+1
                    seg_img[2,int((coords[j][1] - 160) / 10), int(coords[j][0] / 40)] = float((coords[j][0]%40)/40)


            seg_path = img_path.split("/")
            seg_path, img_name = os.path.join(args.root, seg_path[0],seg_path[1], seg_path[2]), seg_path[3]

            seg_path = os.path.join(seg_path, img_name[:-3]+"npy")
            #cv2.imwrite(seg_path, seg_img)
            np.save(seg_path, seg_img)




def generate_json_file(save_dir, json_file, image_set):
    with open(os.path.join(save_dir, json_file), "w") as outfile:
        for json_name in (image_set):
            with open(os.path.join(args.root, json_name)) as infile:
                for line in infile:
                    outfile.write(line)

def generate_label(args):

    #generate_json_file(args.root, "train_val.json", TRAIN_VAL_SET)
    #generate_json_file(args.root, "test.json", TEST_SET)

    #print("generating train_val set...")
    #gen_label_for_json(args, 'train_val')
    print("generating test set...")
    gen_label_for_json(args, 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='D:/TuSimple/test',  help='The root of the Tusimple dataset')
    #parser.add_argument('--savedir', type=str, default='seg_label', help='The root of the Tusimple dataset')
    args = parser.parse_args()

    generate_label(args)
