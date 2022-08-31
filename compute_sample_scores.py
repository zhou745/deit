import os

import torch
import numpy as np
from os import listdir
from os.path import join
from tqdm import tqdm

def reassemble_data(f_path,cls_num,save_path):
    os.makedirs(save_path,exist_ok=True)
    #compute the most similar case class by class
    file_list = listdir(f_path)
    for cls in range(cls_num):
        f_cls_tmp = []
        name_cls_tmp = []

        for file in tqdm(file_list):
            file_path = join(f_path,file)
            state_dict = torch.load(file_path,map_location="cpu")
            feature = state_dict["img"]
            label = state_dict["label"]
            names = np.array(state_dict["name"])

            select_mask = label==cls
            if select_mask.sum().to(torch.float32)>0.5:
                feature_select = feature[select_mask]
                names_select = names[select_mask]
                f_cls_tmp.append(feature_select)
                name_cls_tmp.append(names_select)
        f_cls_all = torch.cat(f_cls_tmp,dim=0)
        name_cls_all = np.concatenate(name_cls_tmp,axis=0)
        save_path_cls = join(save_path,str(cls))
        torch.save({
            "img":f_cls_all,
            "name":name_cls_all
        },save_path_cls)

def compute_sim_matrix(cls,tran_path,val_path,device,norm = True):
    train_path_cls = join(tran_path,str(cls))
    val_path_cls = join(val_path, str(cls))

    dict_train = torch.load(train_path_cls,map_location="cpu")
    dict_val = torch.load(val_path_cls,map_location="cpu")

    train_f = dict_train["img"].to(device)
    val_f = dict_val["img"].to(device)

    train_name = dict_train["name"]
    val_name = dict_val["name"]
    if norm==True:
        train_f = train_f/train_f.norm(dim=-1,keepdim=True)
        val_f = val_f / val_f.norm(dim=-1, keepdim=True)
    sim_train = train_f[:,-1,:]@train_f[:,-1,:].t()
    sim_val = val_f[:,-1,:]@train_f[:,-1,:].t()
    return(sim_train,sim_val,train_name,val_name)

def proecess_scores(scores,names):
    tmp_dict = {}
    scores_max,_ = scores.max(dim=0)
    for idx in range(names.shape[0]):
        name = names[idx].replace("/dataset/train/","")
        tmp_dict.update({name:scores_max[idx].item()})
    return(tmp_dict)

def proecess_scores_v1(scores,names):
    tmp_dict = {}
    scores_max,index = scores.max(dim=1)
    #pre define all the scores
    scores_d = torch.ones((scores.shape[1]),dtype=torch.float32).to(scores_max.device)
    for idx in range(index.shape[0]):
        scores_d[index[idx]] += 1.

    for idx in range(names.shape[0]):
        name = names[idx].replace("/dataset/train/","")
        tmp_dict.update({name:scores_d[idx].item()})
    return(tmp_dict)

def main():
    root_val_path = "./dataset_feature/train_val_noaug"
    train_val_f_path = root_val_path + "/train_ori"
    val_val_f_path = root_val_path + "/val_ori"

    train_val_cls_path = root_val_path + "/train_cls"
    val_val_cls_path = root_val_path + "/val_cls"

    root_path = "./dataset_feature/train_noaug"
    train_f_path = root_path + "/train_ori"
    val_f_path = root_path + "/val_ori"

    train_cls_path = root_path + "/train_cls"
    val_cls_path = root_path + "/val_cls"
    score_cls_path = root_path+"/scores_sample"
    #re assmemble the data
    # reassemble_data(train_f_path,1000,train_cls_path)
    # reassemble_data(val_f_path,1000,val_cls_path)
    #compute the similarity matrix
    score_dict = {}


    device = torch.device("cuda")
    for cls in tqdm(range(1000)):
        with torch.no_grad():
            sim_train_noval, sim_val_noval,train_names, val_names = compute_sim_matrix(cls,train_cls_path,val_cls_path,device)
            # sim_train_val, sim_val_val = compute_sim_matrix(cls, train_val_cls_path, val_val_cls_path,device)
        #scoring the training sample
        tmp_dict = proecess_scores_v1(sim_val_noval,train_names)
        score_dict.update(tmp_dict)
    torch.save(score_dict,score_cls_path)
    print("compute finished")



if __name__=="__main__":
    main()