
import os
def read(pred_folder):
    pred_files = sorted(os.listdir(pred_folder))
    m = 0
    for pred_file in tqdm(pred_files):
        pred_boxes = np.load(os.path.join(pred_folder, pred_file))
        # print(pred_boxes.shape[1])
        if pred_boxes.shape[0] == 0:
            m = m + 1
            print(pred_file)
        
    print(m)

if __name__ == '__main__':
    # vi = Viewer()
    # 文件夹路径
    # from Viewer.viewer.viewer import Viewer
    import numpy as np

    # vi = Viewer()
    # gt_folder = '/mnt/32THHD/lwk/datas/OPV2V/gt_box'  # Ground Truth 框文件夹路径
    # pred_folder = "/mnt/32THHD/lwk/datas/OPV2V/pre_box"  # 预测框文件夹路径
    out_folder = "/mnt/32THHD/lwk/datas/OPV2V/out_final"
    # 计算 Precision 和 Recall
    from tqdm import tqdm
    read(out_folder)

    # box = np.load("/mnt/32THHD/lwk/datas/OPV2V/out_final/pre_4859.npy")
    # print(box.shape[0])

