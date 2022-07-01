from global_names import (A2D2_PATH, sensor_p, abs_)
import pickle, random, cv2, os, json, string, imagesize, copy
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor
%matplotlib inline

with open("bm_ds.pkl", "rb") as f:
    bm_ds = pickle.load(f)

with open(os.path.join(A2D2_PATH, "camera_lidar_semantic/class_list.json"), 'rb') as f:
    class_list = json.load(f)
    
f = lambda x: int(x, 16)
rgb_int = lambda rgb_list: tuple([i for i in map(f, (rgb_list[1:3], rgb_list[3:5], rgb_list[5:7]))])
class_from_rgb = lambda rgb_list: list(class_list.values())[[list((int(i[1:3], 16), int(i[3:5], 16), int(i[5:7], 16))) for i in class_list.keys()].index(list(rgb_list))]
class_list_simplified = {k: (v[:-2] if v[-1] in string.digits else v) for k,v in class_list.items()}


def run_iter(r_id):
    lab_p = abs_(sensor_p(r_id, "label"))
    mask_arr = cv2.cvtColor(cv2.imread(lab_p),cv2.COLOR_BGR2RGB)
    image_bboxes_dict = {i:[] for i in np.unique(list(class_list_simplified.values()))}
    for rgb_list, obj_class in class_list_simplified.items():
        binary_mask = np.where(np.prod(np.where(mask_arr == rgb_int(rgb_list), 255, 0), axis=-1) == 0, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # xmin, xmax, ymin, ymax
        SIZE_LIMIT = 20
        image_bboxes_dict[obj_class] += [i for i in [[c[:,:,0].min(), c[:,:,0].max(), c[:,:,1].min(), c[:,:,1].max()] for i, c in enumerate(contours)] if i[1] - i[0] > SIZE_LIMIT or i[3] - i[2] > SIZE_LIMIT]
        all_bboxes[r_id] = image_bboxes_dict
        
        
all_ids = []
all_bboxes = {}
for i in bm_ds.values():
    all_ids += list(i)
with ThreadPoolExecutor(16) as executor:
     = executor.map(run_iter, all_ids)

file_save = "all_bboxes.pkl"
while os.path.exists(file_save):
    file_save = file_save[:-4] + "_and_another_one" + ".pkl"
    
with open(file_save, "wb") as write_file:
    pickle.dump(results, write_file)