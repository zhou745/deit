import os

import torch
import torch.nn as nn
import numpy as np
from os import listdir
from os.path import join
from tqdm import tqdm
from collections import OrderedDict

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

class classifier(nn.Module):
    def __init__(self,input_dim = 192, output_dim = 1000):

        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim,output_dim)
    def forward(self, input):
        h1 = self.norm(input)
        out = self.head(h1)
        return(out)

def load_classifier(model,ckpt_path):
    state_dict = torch.load(ckpt_path)
    state_dict_model = state_dict['model']
    keys = ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']
    state_dict_new = OrderedDict()
    for key in keys:
        state_dict_new[key] = state_dict_model[key]
    model.load_state_dict(state_dict_new)
    return(model)

def eval_classifer(model, val_f, val_l):
    model.eval()
    pred_val = model(val_f)
    pred_val_i = pred_val.argmax(dim=-1)
    acc = (pred_val_i == val_l).to(torch.float32).mean()
    print("acc is %f" % (acc.item()))

def load_data_val(val_path,device):
    val_f = []
    val_l = []

    for cls in tqdm(range(1000)):

        val_path_cls = join(val_path, str(cls))
        dict_val = torch.load(val_path_cls, map_location="cpu")

        val_f.append(dict_val["img"][:, -1, :])
        val_l.append(cls * torch.ones((val_f[-1].shape[0],), dtype=torch.long))

    # assemble all the features
    val_f_cuda = torch.cat(val_f, dim=0).to(device)
    val_l_cuda = torch.cat(val_l, dim=0).to(device)
    return(val_f_cuda,val_l_cuda)

def load_data_train(train_path,):
    train_f = []
    train_l = []

    for cls in tqdm(range(1000)):
        train_path_cls = join(train_path, str(cls))
        dict_train = torch.load(train_path_cls, map_location="cpu")

        train_f.append(dict_train["img"][:, -1, :])
        train_l.append(cls * torch.ones((train_f[-1].shape[0],), dtype=torch.long))

    # assemble all the features
    split = 5
    train_f_cuda_list = []
    train_l_cuda_list = []

    start = 0
    offset = len(train_f) // split
    for spl in range(split):
        end = start + offset
        train_f_cuda_list.append(torch.cat(train_f[start:end], dim=0))
        train_l_cuda_list.append(torch.cat(train_l[start:end], dim=0))
        start = end
    del (train_f)
    return(train_f_cuda_list,train_l_cuda_list)

def train_classifier(model,optimizer,iter_num,train_path,val_path,scole_path,device):
    #print begin loading all the images
    train_f = []
    scores_f = []
    val_f = []
    train_l = []
    val_l = []
    scores_dict = torch.load(scole_path,map_location="cpu")
    for cls in tqdm(range(1000)):
        train_path_cls = join(train_path, str(cls))
        val_path_cls = join(val_path, str(cls))

        dict_train = torch.load(train_path_cls, map_location="cpu")
        dict_val = torch.load(val_path_cls, map_location="cpu")

        train_f.append(dict_train["img"][:,-1,:])
        val_f.append(dict_val["img"][:,-1,:])
        scores_tmp = torch.zeros((train_f[-1].shape[0],),dtype=torch.float32)
        names_tmp = dict_train["name"]
        for name_id in range(len(names_tmp)):
            scores_tmp[name_id] += scores_dict[names_tmp[name_id].replace("/dataset/train/","")]
        train_l.append(cls*torch.ones((train_f[-1].shape[0],),dtype=torch.long))
        val_l.append(cls*torch.ones((val_f[-1].shape[0],), dtype=torch.long))
        scores_f.append(scores_tmp)
    #assemble all the features
    split = 5
    train_f_cuda_list = []
    train_l_cuda_list = []
    train_s_cuda_list = []
    start = 0
    offset = len(train_f)//split
    for spl in range(split):
        end = start+offset
        train_f_cuda_list.append(torch.cat(train_f[start:end],dim=0))
        train_l_cuda_list.append(torch.cat(train_l[start:end], dim=0))
        train_s_cuda_list.append(torch.cat(scores_f[start:end], dim=0))
        start = end
    del(train_f)
    val_f_cuda = torch.cat(val_f,dim=0).to(device)
    val_l_cuda = torch.cat(val_l,dim=0).to(device)
    #start training
    print("loading finihsed start training")
    model.train()
    for iter in tqdm(range(iter_num)):
        optimizer.zero_grad()
        for idx in range(split):
            imgs = train_f_cuda_list[idx].to(device)
            labels = train_l_cuda_list[idx].to(device)
            scores = train_s_cuda_list[idx].to(device)
            pred = model(imgs)
            loss_all = nn.CrossEntropyLoss(reduction='none')(pred,labels)
            loss = (loss_all*scores).sum()/255000.
            loss.backward()

        optimizer.step()

        if iter%100==0:
            model.eval()
            pred_val = model(val_f_cuda)
            pred_val_i = pred_val.argmax(dim=-1)
            acc= (pred_val_i==val_l_cuda).to(torch.float32).mean()
            print("loss is %f, acc is %f"%(loss.item(),acc.item()))
            torch.save(model.state_dict(),"./score_weight.pth")

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
    path = "./training/train_noaug/checkpoint.pth"

    device = torch.device("cuda")
    #train a classification model
    model = classifier(192,1000)
    optimizer = torch.optim.Adam(params=model.parameters(),lr = 1e-5)
    iter_num= 100000
    load_classifier(model,path)
    model.to(device)
    val_f,val_l = load_data_val(val_cls_path,device)
    eval_classifer(model,val_f,val_l)
    print("finished")
    # train_classifier(model,optimizer,iter_num,train_cls_path,val_cls_path,score_cls_path,device)

    # for cls in tqdm(range(1000)):
    #     with torch.no_grad():
            # sim_train_noval, sim_val_noval,train_names, val_names = compute_sim_matrix(cls,train_cls_path,val_cls_path,device)
            # sim_train_val, sim_val_val = compute_sim_matrix(cls, train_val_cls_path, val_val_cls_path,device)
        #scoring the training sample
        # tmp_dict = proecess_scores_v1(sim_val_noval,train_names)
        # score_dict.update(tmp_dict)
    # torch.save(score_dict,score_cls_path)
    print("compute finished")



if __name__=="__main__":
    main()