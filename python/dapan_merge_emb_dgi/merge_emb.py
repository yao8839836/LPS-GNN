import os
import sys

#merge_lines = []

date = sys.argv[1]

qq_emb_files = os.listdir("dapan/qq/dapan_qq_emb/")
print(qq_emb_files)

wechat_emb_files = os.listdir("dapan/wechat/dapan_wechat_emb/")
print(wechat_emb_files)

os.system("rm -r dapan/dapan_merge_emb.csv")

for emb_file in qq_emb_files:
    emb_lines = []
    with open("dapan/qq/dapan_qq_emb/" + emb_file,'r') as f:
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            line = line.strip()
            temp = line.split("\01")
            new_vector_list = temp[1].split(" ")
            new_vector_str = "\01".join(new_vector_list)
            emb_lines.append(date + "\01" + temp[0] + "\01" + new_vector_str + "\n")

    with open("dapan/dapan_merge_emb.csv" , "a") as f:

        f.writelines(emb_lines)

for emb_file in wechat_emb_files:
    emb_lines = []
    with open("dapan/wechat/dapan_wechat_emb/" + emb_file,'r') as f:
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            line = line.strip()
            temp = line.split("\01")
            new_vector_list = temp[1].split(" ")
            new_vector_str = "\01".join(new_vector_list)
            emb_lines.append(date + "\01" + temp[0] + "\01" + new_vector_str + "\n")

    with open("dapan/dapan_merge_emb.csv" , "a") as f:

        f.writelines(emb_lines)