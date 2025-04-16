import numpy as np


weight_dict = {}

with open("/dapan/qq/dapan_sns_edges_qq_community", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.split("\01")
        #weight_array.append(int(tmp[2]))
        w = int(tmp[2])
        src_id = int(tmp[0])
        dst_id = int(tmp[1])
        if src_id not in weight_dict:
            weight_dict[src_id] = [w]
        else:
            src_array = weight_dict[src_id]
            src_array.append(w)
            weight_dict[src_id] = src_array

        if dst_id not in weight_dict:
            weight_dict[dst_id] = [w]
        else:
            dst_array = weight_dict[dst_id]
            dst_array.append(w)
            weight_dict[dst_id] = dst_array

weight_threshold = {}

for id in weight_dict:
    w_array = weight_dict[id]
    w_array = np.array(w_array)
    sorted_array = np.sort(w_array)
    print(np.max(sorted_array), np.min(sorted_array))
    threshold_index = int(0.9 * len(w_array))
    threshold = sorted_array[threshold_index]
    weight_threshold[id] = threshold


keep_lines = []

for line in lines:
    tmp = line.split("\01")
    w = int(tmp[2])
    if w >= weight_threshold[int(tmp[0])] and w >= weight_threshold[int(tmp[1])]:
        keep_lines.append(line)
        
with open("/dapan/qq/dapan_sns_edges_qq_community_keep", "w") as f:
    f.writelines(keep_lines)


weight_dict = {}

with open("/dapan/wechat/dapan_sns_edges_wechat_community", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.split("\01")
        #weight_array.append(int(tmp[2]))
        w = int(tmp[2])
        src_id = int(tmp[0])
        dst_id = int(tmp[1])
        if src_id not in weight_dict:
            weight_dict[src_id] = [w]
        else:
            src_array = weight_dict[src_id]
            src_array.append(w)
            weight_dict[src_id] = src_array

        if dst_id not in weight_dict:
            weight_dict[dst_id] = [w]
        else:
            dst_array = weight_dict[dst_id]
            dst_array.append(w)
            weight_dict[dst_id] = dst_array

weight_threshold = {}

for id in weight_dict:
    w_array = weight_dict[id]
    w_array = np.array(w_array)
    sorted_array = np.sort(w_array)
    print(np.max(sorted_array), np.min(sorted_array))
    threshold_index = int(0.9 * len(w_array))
    threshold = sorted_array[threshold_index]
    weight_threshold[id] = threshold


keep_lines = []

for line in lines:
    tmp = line.split("\01")
    w = int(tmp[2])
    if w >= weight_threshold[int(tmp[0])] and w >= weight_threshold[int(tmp[1])]:
        keep_lines.append(line)
        
with open("/dapan/wechat/dapan_sns_edges_wechat_community_keep", "w") as f:
    f.writelines(keep_lines)

