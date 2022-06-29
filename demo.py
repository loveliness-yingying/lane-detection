import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor

color_list = [(0, 0, 0), (0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,255),(0,255,255)]
h_samples = [115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235, 245, 255, 265,
                     275, 285, 295, 305, 315, 325, 335, 345, 355, 365, 375, 385, 395, 405, 415,425, 435, 445, 455, 465,
                     475, 485, 495, 505, 515, 525, 535, 545, 555, 565, 575, 585]
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 48
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False) # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        #transforms.Resize((448, 512)),
        transforms.Resize((384, 640)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        splits = ['test_signal.txt']
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, split), img_transform = img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        splits = ['test_signal.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError
    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs
            with torch.no_grad():
                out = net(imgs)
                out = out[2]
            out = out[0]

            cls = out[0:5,:,:]
            offset = out[5,:,:]
            cls = cls.numpy()
            offset = offset.numpy()
            cls = np.argmax(cls,axis=0)

            #label_name = names[0][:-3] + 'npy'
            #label = np.load(os.path.join(cfg.data_root,label_name))

            # import pdb; pdb.set_trace()
            vis = cv2.imread(os.path.join(cfg.data_root,names[0]))

            """
            for i in range(56):
                for j in range(32):
                    if label[0,i,j]!=0:
                        offset_val = label[2,i,j]
                        cls_num = int(label[1,i,j])
                        ppp = (int(j*40+offset_val*40),int(i*10+160))
                        cv2.circle(vis,ppp,7,color_list[cls_num],-1)
            """


            for i in range(48):
                for j in range(40):
                    if cls[i,j]!=0:
                        cls_num = cls[i,j]
                        offset_val = offset[i,j]
                        ppp = (int(j*41+offset_val*41),int(h_samples[i]))
                        cv2.circle(vis,ppp,7,color_list[cls_num],-1)


        cv2.imwrite('test.jpg',vis)
