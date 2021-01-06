import os
import json
import xml.etree.ElementTree as ET
import numpy as np

threshold = 0.75
GT_size = {}
outdict = {}

def GT_labels(path):
    # print(threshold)
    # print(path)
    
    dic = {}
    _, GT_dirs, _ = next(os.walk(path))
    # print(GT_dirs)

    for cur_dir in GT_dirs:
        cur_path = os.path.join(path, cur_dir)
        # print(cur_path)
        cur_size = 0
        for filename in os.listdir(cur_path):
            # print(filename)
            tree = ET.parse(os.path.join(cur_path, filename))
            file = tree.find('filename').text
            bbox = tree.findall('object')[0].find('bndbox')
            list = [int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text)]
            dic[file] = list
            cur_size += 1
        GT_size[cur_dir.split('-')[-1]] = cur_size
    return dic

def res_labels(path):
    dic = {}
    for filename in os.listdir(path):
        if filename.split('.')[-1] == 'manifest':
            continue
        file = filename.split('.')[0].split('-')[0] + '.jpg'

        cur_list = []
        # print(path, filename)
        with open(os.path.join(path, filename), "r") as f: 
            tree = json.load(f)
            # print(path, filename)
            if tree is not None:
                if 'detection_boxes' not in tree:
                    continue
                detection = tree["detection_boxes"]
                confidence = tree["detection_scores"]
                for i, x in enumerate(detection):
                    tmp = {"bbox": [float(x[1]), float(x[0]), float(x[3]), float(x[2])], "confidence": confidence[i]}
                    cur_list.append(tmp)
        dic[file] = cur_list
    return dic

def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    area = w * h
    score = area / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                    + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - area)
    return score

def takeconf(elem):
    return elem["confidence"]

def work(cur_dir, allgt, gtdict, testdict):
    maxscore, maxpos = 0, -1
    list = []
    sumiou = 0.0
    times = 0
    for img, _ in testdict.items():
        testboxes = testdict[img]
        gtbox = gtdict[img]
        # print(img,testboxes,gtbox)
        for i, testbox in enumerate(testboxes):
            detection = testbox["bbox"]
            score = iou(detection, gtbox)
            if score > maxscore:
                maxscore = score
                maxpos = i
        # print(maxscore)
        for i, x in enumerate(testboxes):
            if (i == maxpos) and (maxscore > threshold):
                list.append({"confidence": x["confidence"], "tp": 1})
                sumiou += maxscore
                times += 1
            else:
                list.append({"confidence": x["confidence"], "tp": 0})

    sumiou = 0 if times == 0 else sumiou / times
    # print("mIoU = %f" % round(sumiou, 4), end=' ')
    list.sort(reverse=True, key=takeconf)
    Plist, Rlist = [], []
    sumTP = 0
    for i, x in enumerate(list):
        if x["tp"] == 1:
            sumTP += 1
        Plist.append(sumTP / (i + 1))
        Rlist.append(sumTP / allgt)
    pnp = np.array(Plist)
    rnp = np.array(Rlist)
    ap = 0
    pre = 0
    
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rnp >= t) == 0:
            pre = 0
        else:
            pre = np.max(pnp[rnp >= t])
        # print(pre)
        ap += pre / 11
    # print("AP = %f" % round(ap, 4), end=' ')
    ap_voc = voc_ap(rnp, pnp)
    outdict[cur_dir] = {"mIoU": round(sumiou, 4), "AP": round(ap, 4), "AP-voc": ap_voc}

def voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    # print(mpre)
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
 
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    # print(ap)
    return ap


if __name__ == "__main__":
    print("Begin!")
    root = os.getcwd()
    GT_root = os.path.join(root,'GT_labels')
    res_root = os.path.join(root,'results')
    out_root = os.path.join(root,'output')
    # print(GT_root)
    
    _, dirs, _ = next(os.walk(res_root))
    
    GT_dict = GT_labels(GT_root)
    # print(GT_dict)
    # print(GT_size)

    

    for cur_dir in dirs:
        cur_path = os.path.join(res_root, cur_dir)

        cur_GT_size = 0
        if cur_dir.split('-')[-1] == 'sui':
            cur_GT_size = GT_size['sui']
        elif cur_dir.split('-')[-1] == 'tuo':
            cur_GT_size = GT_size['tuo']
        elif cur_dir.split('-')[-1] == 'yuan':
            cur_GT_size = GT_size['yuan']
        else: 
            for key, value in GT_size.items():
                cur_GT_size += value
        # print(cur_path, cur_GT_size)

        res_dict = res_labels(cur_path)
        # print(res_dict)
        # break
        # print(cur_dir + ":", end=' ')
        work(cur_dir, cur_GT_size, GT_dict, res_dict)

    outlist = sorted(outdict.items(), key=lambda x:x[0])
    for key in outlist:
        print(key[0], key[1])
    with open(out_root+"/results.txt", "w") as f: 
        json.dump(outlist, f)

    with open(out_root+"/res.txt", "w") as f: 
        for key in outlist:
            f.write(str(key[0])+ " " + str(key[1]))
            f.write("\n")