#coding:utf-8
import pandas as pd
import re
from models.albert_for_ner import AlbertCrfForNer
from models.bert_for_ner import BertCrfForNer
from models.electra_for_ner import ElectraCrfForNer
from processors.utils_ner import CNerTokenizer, get_entities
import torch
from processors.ner_seq import CnerProcessor
from fix_by_rule.fix_by_rule1 import fix
from tqdm import tqdm
import flask
from flask import request
import argparse
import numpy as np
import time
class AnnData():
    def __init__(self):
        ## 设置参数
        self.model_type="electra"
        self.model_path="bushu/server_model/"+self.model_type.lower()
        self.corpus_path="datasets/corpus_162w_foolnltk/corpus.txt"
        self.use_GPU = False
        self.batch_size=32
        print("完成初始化参数")
        ## 载入模型
        if self.model_type =="bert":
            self.model = BertCrfForNer.from_pretrained(self.model_path)
        elif self.model_type =="albert":
            self.model = AlbertCrfForNer.from_pretrained(self.model_path)
        else:
            self.model = ElectraCrfForNer.from_pretrained(self.model_path)

        self.tokenizer=CNerTokenizer.from_pretrained(self.model_path)
        self.model=self.model.eval()
        print("完成模型载入")

        if self.use_GPU:
            self.model=self.model.cuda()
        self.processor=CnerProcessor()
        label_list = self.processor.get_labels()
        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.label2id = {label: i for i, label in enumerate(label_list)}


    def strQ2B(self,ustring):
        """全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code=ord(uchar)
            if inside_code == 12288:                              #全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
                inside_code -= 65248

            rstring += chr(inside_code)
        return rstring

    def AnnData(self, Max_cut=128):
        """
            标注数据
        """
        batch_size = self.batch_size
        use_GPU = self.use_GPU
        model=self.model
        tokenizer=self.tokenizer
        sent_list=[]#一个batch的句子
        to_write=[]
        num_tokens_writed=0
        with open (self.corpus_path+".ann","w",encoding="utf-8") as g:
            with open(self.corpus_path,encoding="utf-8") as f:
                for sent in f:
                    if len(sent_list)<batch_size:
                        if sent != "\n":
                            sent_list.append(sent)
                    else:#够一个batch
                        input_ids_=[]#每个句子字符转ID
                        tokens_=[]#每个句子的字符
                        inputs_mask_ =[]
                        len_list_=[]
                        for i,sent in enumerate(sent_list):
                            tokens=tokenizer.tokenize(sent.lower())
                            if len(tokens) > Max_cut-2:
                                tokens = tokens[:Max_cut-2]
                            N_tokens=len(tokens)
                            tokens=["[CLS]"]+tokens+["[SEP]"]+(Max_cut-2-N_tokens)*["[PAD]"]
                            inputs_mask_.append([0]+[1]*N_tokens+[0]+(Max_cut-2-N_tokens)*[0])
                            input_ids_.append(tokenizer.convert_tokens_to_ids(tokens))
                            tokens_.append(tokens)
                            len_list_.append(N_tokens)
                        ## [N_sents,Max_cut]
                        input_ids_ = torch.tensor(input_ids_, dtype=torch.long)#所有句子的ID张量
                        if use_GPU:
                            input_ids_=input_ids_.cuda()

                        inputs = {"input_ids": input_ids_}
                        logits=model(**inputs)
                        decodes=model.crf.decode(logits[0]).cpu().numpy().tolist()#一个batch解码
                        tags_=decodes[0]

                        for k in range(len(tokens_)):
                            TOKENs = tokens_[k][1:1+len_list_[k]]
                            LABELs = [self.id2label[tagid] for tagid in tags_[k][1:1+len_list_[k]]]
                            token_label_pairs=[TOKENs[j]+" "+LABELs[j]+"\n"  for j in range(len(TOKENs))]+["\n"]
                            to_write.extend(token_label_pairs)
                            if len(to_write)>10000:
                                g.writelines(to_write)
                                num_tokens_writed+=len(to_write)
                                print(f"已经写入{self.corpus_path}.ann文件{num_tokens_writed}字")
                                to_write=[]

                        sent_list = []


if __name__ == "__main__":
    qidianNER=AnnData()
    strat=time.time()
    res=qidianNER.AnnData()


