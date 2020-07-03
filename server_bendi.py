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
class qdNER():
    def __init__(self):
        ## 设置参数
        self.model_type="electra"
        self.model_path="bushu/server_model/"+self.model_type.lower()
        self.use_GPU = False
        self.batch_size=64
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
            self.model=model.cuda()
        self.processor=CnerProcessor()
        label_list = self.processor.get_labels()
        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.label2id = {label: i for i, label in enumerate(label_list)}

    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

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

    def predict_paragraphs_with_offset_allpad(self, paragraphs, Max_cut=128):
        """
            paragraph:多篇文章，多句当做一个batch预测
        """
        batch_size = self.batch_size
        use_GPU = self.use_GPU
        model=self.model
        tokenizer=self.tokenizer
        sent_list=[]#每篇文章的句子
        num_sents=[]#前N篇文章的句子数
        sents_offset=[]

        for paragraph in paragraphs:
            # paragraph = self.strQ2B(paragraph)
            new_sents=self.cut_sent(paragraph)
            sent_list.extend(new_sents)
            num_sents.append(len(sent_list))
            num_sents_numtoken=[len(sent) for sent in new_sents]
            s_offset=[0]+np.cumsum(num_sents_numtoken).tolist()[:-1]
            sents_offset.extend(s_offset)

        input_ids_=[]#每个句子字符转ID
        tokens_=[]#每个句子的字符
        inputs_mask_ =[]

        for i,sent in enumerate(sent_list):
            tokens=tokenizer.tokenize(sent.lower())
            if len(tokens) > Max_cut-2:
                tokens = tokens[:Max_cut-2]
            N_tokens=len(tokens)
            tokens=["[CLS]"]+tokens+["[SEP]"]+(Max_cut-2-N_tokens)*["[PAD]"]
            inputs_mask_.append([0]+[1]*N_tokens+[0]+(Max_cut-2-N_tokens)*[0])
            input_ids_.append(tokenizer.convert_tokens_to_ids(tokens))
            tokens_.append(tokens)

        ## [N_sents,Max_cut]
        input_ids_ = torch.tensor(input_ids_, dtype=torch.long)#所有句子的ID张量
        if use_GPU:
            input_ids_=input_ids_.cuda()

        tags = []
        N_sents=len(sent_list)
        for i in tqdm(range(0,N_sents,batch_size),desc=f"Sentences_num={N_sents},Batch_size={batch_size}"):#按batch_size预测
            input_ids=input_ids_[i:min(i+batch_size,N_sents)]
            inputs = {"input_ids": input_ids}
            logits=model(**inputs)
            decodes=model.crf.decode(logits[0]).cpu().numpy().tolist()#一个batch解码
            tags.extend(decodes[0])

        results_ = []
        for i,preds in enumerate(tags):# 逐个句子
            tokens=tokens_[i]
            preds = preds[1:-1]  # [CLS]XXXX[SEP]
            label_entities = get_entities(preds, self.id2label, "bio")
            tokens=tokens[1:-1]
            entity_1sent={}#一句话中的实体，以及在全文的位置
            for entity in label_entities:
                s=entity[1]
                e=entity[2]+1
                entity_str="".join(tokens[entity[1]:entity[2]+1])
                if entity_str not in entity_1sent and "[PAD]" not in entity_str:
                    entity_1sent["".join(tokens[s:e])]=(s+sents_offset[i],e+sents_offset[i])#计算在全文中的位置
            results_.append(entity_1sent)
        num=0
        results=[]
        for i in num_sents:
            para_entitys={}
            para_entitys_=results_[num:i]
            for dict_ in para_entitys_:
                for k in dict_:
                    if k not in para_entitys:
                        para_entitys[k]=dict_[k]
            num=i
            results.append(para_entitys)
        return results

    def predict(self,paragraphs):
        results=self.predict_paragraphs_with_offset_allpad(paragraphs=paragraphs)
        ## torch释放未占用内存：Pytorch已经可以自动回收我们不用的显存，类似于python的引用机制，
        ## 当某一内存内的数据不再有任何变量引用时，这部分的内存便会被释放。但有一点需要注意，
        ## 当我们有一部分显存不再使用的时候，这部分释放的显存通过Nvidia-smi命令是看不到的。
        if self.use_GPU:
            torch.cuda.empty_cache()
        return results



if __name__ == "__main__":
    qidianNER=qdNER()
    strat=time.time()
    articles=["君正集团近日发布公告称，华泰保险集团已收到银保监会《关于华泰保险集团股份有限公司变更股东的批复》，同意君正集团、内蒙古君正化工有限责任公司分别将持有的华泰保险集团4.72亿股股份、1.43亿股股份转让给安达天平再保险有限公司（下称安达天平），转让完成后，安达天平对华泰保险集团的持股比例将达到25.0823%",
                                        "由“软银集团”(SoftBank Group Corp.)支持的保险初创公司Lemonade周四表示，该公司计划通过在美国的首次公开募股(IPO)筹集高达2.86亿美元的资金。",
                                        "纳斯达克总法律顾问办公室已通知公司，公司股票将于2020年6月29日开盘时停牌，纳斯达克将在所有上诉期限届满后提交退市通知。事实上，在近一个月中，瑞幸咖啡已经收到纳斯达克两次退市通知，目前瑞幸咖啡已经放弃了自我挣扎 。"]*128
    res=qidianNER.predict(paragraphs=articles)
    end=time.time()
    delta_time=end-strat
    num_tokens=sum([len(article) for article in articles])
    print(f"Speed={num_tokens/delta_time}字/秒")
    ## Speed=3237.6字/秒 -- batch_size=32
    ## Speed=3237.6字/秒 -- batch_size=64

    # for ner in res:
        #     print(ner)


