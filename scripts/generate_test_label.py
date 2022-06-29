import json
import numpy as np
import os
import argparse



def gen_label_for_json(args, image_set):

    out_path = os.path.join(args.root,  "{}.json".format(image_set))
    outfp = open(out_path, 'w')
    with open(os.path.join(args.root, 'test.txt')) as content_file:
        for line in content_file:
            raw_file_path = line[:-1]
            label_path = raw_file_path[:-3] + 'npy'

            tmp_dict = {}
            tmp_dict['lanes'] = generate_lines(args,label_path)
            tmp_dict['h_samples'] = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
                                     270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
                                     430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580,
                                     590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
            tmp_dict['raw_file'] = raw_file_path
            json_str = json.dumps(tmp_dict)
            outfp.write(json_str + '\n')
    outfp.close()


def generate_lines(args,content_path):
    label = np.load(os.path.join(args.root,content_path))
    cls = label[1]
    offset = label[2]
    lanes = [[-2 for j in range(56)] for i in range(6)]
    for i in range(56):
        for j in range(32):
            if cls[i, j] == 1.0:
                lanes[0][i] = int(j * 40 + offset[i, j] * 40)
            if cls[i, j] == 2.0:
                lanes[1][i] = int(j * 40 + offset[i, j] * 40)
            if cls[i, j] == 3.0:
                lanes[2][i] = int(j * 40 + offset[i, j] * 40)
            if cls[i, j] == 4.0:
                lanes[3][i] = int(j * 40 + offset[i, j] * 40)
            if cls[i, j] == 5.0:
                lanes[4][i] = int(j * 40 + offset[i, j] * 40)
            if cls[i, j] == 6.0:
                lanes[5][i] = int(j * 40 + offset[i, j] * 40)
    return lanes



def generate_label(args):
    print("generating test set...")
    gen_label_for_json(args, 'test_label')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='D:/TuSimple/test',  help='The root of the Tusimple dataset')
    args = parser.parse_args()
    generate_label(args)
