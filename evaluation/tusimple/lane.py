import numpy as np
from sklearn.linear_model import LinearRegression
import json


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        #pred = np.array([p if p >= 0 else -100 for p in pred])
        pred_list = []
        gt = np.array([g if g >= 0 else -100 for g in gt])
        for i in range(len(pred)):
            if pred[i]<=0 or gt[i]==-100:
                pred_list.append(-100)
            else:
                pred_list.append(pred[i])
        pred = np.array(pred_list)
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        pred_new = []
        gt_new = []
        threshs_new = []
        for i in range(len(angles)):
            if angles[i]!=0:
                pred_new.append(pred[i])
                gt_new.append(gt[i])
                threshs_new.append(threshs[i])
        line_accs = []
        for x_preds, x_gts, thresh in zip(pred_new, gt_new, threshs_new):
            acc = LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh)
            line_accs.append(acc)
        s = sum(line_accs)
        return s / len(line_accs)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy = 0.0
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
        num = len(gts)
        # the first return parameter is the default ranking parameter
        return json.dumps(
            [{'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'}])


if __name__ == '__main__':
    import sys
    try:
        if len(sys.argv) != 3:
            raise Exception('Invalid input arguments')
        print(LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(e.message)
        sys.exit(e.message)
