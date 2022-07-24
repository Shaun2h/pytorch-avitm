import csv
import torch
import os
import random
import json
from torch.utils.data import Dataset, DataLoader

class indo_dataset_class(Dataset):
    def __init__(self, idxes, data,label,backref_dict,tokeniser,device):
        self.idxes = idxes
        self.data = data
        self.label = label
        self.backref_dict = backref_dict
        self.device = device
        self.tokenizer = tokeniser
        
    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, idx):
        data = self.data[self.idxes[idx]]
        label = self.label[self.idxes[idx]]
        textlist = [data]
        with torch.no_grad():
            tokenized_data = self.tokenizer(textlist, return_tensors="pt", padding="max_length", truncation=True)
            tokenized_data["input_ids"] = tokenized_data["input_ids"].squeeze()
            tokenized_data['token_type_ids'] = tokenized_data['token_type_ids'].squeeze()
            tokenized_data['attention_mask'] = tokenized_data['attention_mask'].squeeze()
        tokenized_data.to(self.device)
        return tokenized_data, label,idx, data
    
        
    def backref(self,targetidx):
        return self.data[self.idxes[int(targetidx)]], self.backref_dict[self.data[self.idxes[int(targetidx)]]][0]




class dataset_class_PHEME_tree(Dataset):
    def __init__(self, data,tokeniser,device,targetdumpfile):
        # 0=noise, 1 = rumour
        with open(targetdumpfile,"rb") as dumpfile:
            loaded_threads = json.load(dumpfile)
        self.tokenizer = tokeniser
        self.allthreads = {}
        self.rootitems = []
        self.device = device
        for thread in loaded_threads:
            threadtextlist,tree,rootlabel,source_id = thread
            if str(source_id) in data:
                self.allthreads[source_id] = thread
                self.rootitems.append(source_id)
        
    def __len__(self):
        return len(self.rootitems)

    def __getitem__(self, idx):
        targettree = self.rootitems[idx]
        thread = self.allthreads[targettree]
        threadtextlist,tree,rootlabel,source_id = thread
        textlist = ""
        for item in threadtextlist:
            textlist = textlist+ item[0]+" [SEP]"
        textlist = textlist[:-6] # remove last sep.
        with torch.no_grad():
            tokenized_data = self.tokenizer(textlist, return_tensors="pt", padding="max_length", truncation=True)
            tokenized_data["input_ids"] = tokenized_data["input_ids"].squeeze()
            tokenized_data['token_type_ids'] = tokenized_data['token_type_ids'].squeeze()
            tokenized_data['attention_mask'] = tokenized_data['attention_mask'].squeeze()
        tokenized_data.to(self.device)
        return tokenized_data, rootlabel[0],idx, " ".join(textlist)
        
    def backref(self,idx):
        return self.allthreads[self.rootitems[idx]], self.rootitems[idx]



class dataset_class_PHEME_root_only_tree_inp(Dataset):
    # similar to the above version, but only outputs roots from an input tree.
    def __init__(self, data,tokeniser,device,targetdumpfile,note=""):
        # 0=noise, 1 = rumour
        with open(targetdumpfile,"rb") as dumpfile:
            loaded_threads = json.load(dumpfile)
        self.tokenizer = tokeniser
        self.allthreads = {}
        self.rootitems = []
        self.device = device
        # print(len(data))
        self.labeldict = {}
        rootlabelcounter = 0
        for thread in loaded_threads:
            threadtextlist,tree,rootlabel,source_id = thread
            if not rootlabel[2] in self.labeldict:
                self.labeldict[rootlabel[2]] = rootlabelcounter
                rootlabelcounter+=1
            if str(source_id) in data:
                self.allthreads[source_id] = thread
                self.rootitems.append(source_id)
        vocab = self.tokenizer.get_vocab()
        print(note,self.labeldict)
        self.inp_size = max(list({vocab[z]:z for z in vocab}.keys()))
        
    def __len__(self):
        return len(self.rootitems)

    def __getitem__(self, idx):
        targettree = self.rootitems[idx]
        thread = self.allthreads[targettree]
        threadtextlist,tree,rootlabel,source_id = thread
        # print(threadtextlist)
        for searcher in threadtextlist:
            if searcher[1]==source_id:
                textlist=searcher[0]
                break
        with torch.no_grad():
            tokenized_data = self.tokenizer(textlist, return_tensors="pt", truncation=True)
            tokenized_data["input_ids"] = tokenized_data["input_ids"].squeeze()
            tokenized_data['token_type_ids'] = tokenized_data['token_type_ids'].squeeze()
            tokenized_data['attention_mask'] = tokenized_data['attention_mask'].squeeze()
        returndat = torch.zeros([self.inp_size])
        for i in tokenized_data["input_ids"]:
            returndat[i] = returndat[i] + 1
        # tokenized_data.to(self.device)
        # print(self.labeldict[rootlabel[2]])
        # print(idx)
        # print("".join(textlist))
        # input()
        return returndat[1:].to(self.device), self.labeldict[rootlabel[2]],idx, "".join(textlist)
        
    def backref(self,idx):
        return self.allthreads[self.rootitems[idx]], self.rootitems[idx], self.allthreads[self.rootitems[idx]][-1]


class dataset_class_PHEME_direct(Dataset):
    # takes in tweets DIRECTLY. no tree structure is retained.
    def __init__(self, data,labeldict,tokeniser,device,note=""):
        # 0=noise, 1 = rumour
        
        self.tokenizer = tokeniser
        self.device = device
        # print(len(data))
        self.data = data
        vocab = self.tokenizer.get_vocab()
        self.labeldict = labeldict
        print(note,self.labeldict)
        self.rootitems = [] # save the src tweet.
        for i in data:
            self.rootitems.append(i[-1])
        
        
        self.inp_size = max(list({vocab[z]:z for z in vocab}.keys()))
        
    def __len__(self):
        return len(self.rootitems)

    def __getitem__(self, idx):
        text,selfidstr,selfid,time,event,srcid = self.data[idx]
        event = event.split("-")[0]
        # print(threadtextlist)      
        
        with torch.no_grad():
            tokenized_data = self.tokenizer(text, return_tensors="pt", truncation=True)
            tokenized_data["input_ids"] = tokenized_data["input_ids"].squeeze()
            tokenized_data['token_type_ids'] = tokenized_data['token_type_ids'].squeeze()
            tokenized_data['attention_mask'] = tokenized_data['attention_mask'].squeeze()
        returndat = torch.zeros([self.inp_size])
        for i in tokenized_data["input_ids"]:
            returndat[i-1] = returndat[i-1] + 1
        # print(returndat[1:].shape,self.inp_size)
        # print(returndat)
        # tokenized_data.to(self.device)
        # print(text)
        # print(idx)
        # print(self.labeldict[event])
        # input()
        return returndat[1:].to(self.device), self.labeldict[event],idx, text
        
    def backref(self,idx):
        return self.data[idx], self.rootitems[idx],self.data[idx][2]
