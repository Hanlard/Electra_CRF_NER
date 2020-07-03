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


app = flask.Flask(__name__)

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def strQ2B(ustring):
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

def predict_paragraphs(args, model, tokenizer, paragraphs, batch_size=64,use_GPU=True):
    """
        paragraph:多篇文章，多句当做一个batch预测
    """
    sent_list=[]#每篇文章的句子
    num_sents=[]#每篇文章的句子数
    for paragraph in paragraphs:
        # paragraph = strQ2B(paragraph)
        sent_list.extend(cut_sent(paragraph))
        num_sents.append(len(sent_list))

    input_ids_=[]#每个句子字符转ID
    tokens_=[]#每个句子的字符
    for sent in sent_list:
        tokens=tokenizer.tokenize(sent)
        if len(tokens) > 126:
            tokens = tokens[:126]
        tokens=["[CLS]"]+tokens+["[SEP]"]+(126-len(tokens))*["[PAD]"]
        input_ids_.append(tokenizer.convert_tokens_to_ids(tokens))
        tokens_.append(tokens)

    ## [N_sents,128]
    input_ids_ = torch.tensor(input_ids_, dtype=torch.long)#所有句子的ID张量
    if use_GPU:
        input_ids_=input_ids_.cuda()
        model.cuda()

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
        label_entities = get_entities(preds, args.id2label, args.markup)
        tokens=tokens[1:-1]
        entity_1sent=[]
        for entity in label_entities:
            if "".join(tokens[entity[1]:entity[2]+1]) not in entity_1sent:
                entity_1sent.append("".join(tokens[entity[1]:entity[2]+1])+" ")
        results_.append("".join(entity_1sent))

    num=0
    results=[]
    for i in num_sents:
        para_entitys="".join(results_[num:i])
        num=i
        results.append(para_entitys)

    return results

def predict_paragraphs_with_offset(args, model, tokenizer, paragraphs):
    """
        paragraph:多篇文章，多句当做一个batch预测
    """
    batch_size = args.batch_size
    use_GPU = args.use_GPU

    sent_list=[]#每篇文章的句子
    num_sents=[]#前N篇文章的句子数
    sents_offset=[]
    for paragraph in paragraphs:
        # paragraph = strQ2B(paragraph)
        new_sents=cut_sent(paragraph)
        sent_list.extend(new_sents)
        num_sents.append(len(sent_list))
        num_sents_numtoken=[len(sent) for sent in new_sents]
        s_offset=[0]+np.cumsum(num_sents_numtoken).tolist()[:-1]
        sents_offset.extend(s_offset)
    batchs_input=[]
    batch=[]
    batch_Maxlen=0
    tokens_=[]
    for i,sent in enumerate(sent_list):
        tokens=tokenizer.tokenize(sent.lower())
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        tokens_.append(tokens)
        N_tokens=len(tokens)
        if N_tokens>batch_Maxlen:
            batch_Maxlen=N_tokens
        masks=[1]*N_tokens
        batch.append([tokens,masks])
        if (i+1) % batch_size == 0:
            batch_input_ids = torch.tensor([
                tokenizer.convert_tokens_to_ids(item[0]+["PAD"]*(batch_Maxlen-len(item[0])))
                for item in batch], dtype=torch.long)
            batch_input_masks = torch.tensor([item[1]+[0]*(batch_Maxlen-len(item[0])) for item in batch],dtype=torch.uint8)
            batchs_input.append((batch_input_ids,batch_input_masks))
            batch = []
            batch_Maxlen = 0
    tags = []
    N_sents=len(sent_list)
    for i,(input_ids,inputs_mask) in enumerate(tqdm(batchs_input,desc=f"Sentences_num={N_sents},Batch_size={batch_size}")):#按batch_size预测
        if use_GPU:
            input_ids = input_ids.cuda()
            inputs_mask = inputs_mask.cuda()
        inputs = {"input_ids": input_ids,"attention_mask":inputs_mask}
        logits=model(**inputs)
        decodes=model.crf.decode(logits[0]).cpu().numpy().tolist()#一个batch解码
        tags.extend(decodes[0])
    results_ = []
    for i,preds in enumerate(tags):# 逐个句子
        tokens=tokens_[i]
        preds = preds[1:-1]  # [CLS]XXXX[SEP]
        label_entities = get_entities(preds, args.id2label, "bio")
        tokens=tokens[1:-1]
        entity_1sent={}#一句话中的实体，以及在全文的位置
        for entity in label_entities:
            s=entity[1]
            e=entity[2]+1
            if "".join(tokens[s:e]) not in entity_1sent:
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

def predict_paragraphs_with_offset_allpad(args, model, tokenizer, paragraphs,Max_cut=128):
    """
        paragraph:多篇文章，多句当做一个batch预测
    """
    batch_size = args.batch_size
    use_GPU = args.use_GPU

    sent_list=[]#每篇文章的句子
    num_sents=[]#前N篇文章的句子数
    sents_offset=[]

    for paragraph in paragraphs:
        # paragraph = strQ2B(paragraph)
        new_sents=cut_sent(paragraph)
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
        label_entities = get_entities(preds, args.id2label, "bio")
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

def load_model(args):
    global tokenizer,model,processor,label_list
    model_saved = args.model_path
    use_GPU = args.use_GPU
    model_type = args.model_type
    if model_type =="bert":
        model = BertCrfForNer.from_pretrained(model_saved)
    elif model_type =="albert":
        model = AlbertCrfForNer.from_pretrained(model_saved)
    else:
        model = ElectraCrfForNer.from_pretrained(model_saved)

    tokenizer=CNerTokenizer.from_pretrained(model_saved)
    model.eval()
    if use_GPU:
        model.cuda()
        # model = torch.nn.DataParallel(model)
    processor=CnerProcessor()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}



@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == 'POST':
        paragraphs=request.form.getlist('data')
        # results=predict_paragraphs(args, model, tokenizer, paragraphs)
        results=predict_paragraphs_with_offset_allpad(args, model, tokenizer, paragraphs)
        data['predictions'] = results
        data["success"] = True
        torch.cuda.empty_cache()
    return flask.jsonify(data)

def prepare_data(path="未标注测试数据/结果输出-原始数据.xlsx"):
    Df = pd.read_excel(path, index_col=0)
    Df["albert_crf"]=""
    paragraphs=Df['content'].to_list()[0:100]
    return paragraphs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="electra", type=str, help="模型类型")
    parser.add_argument("--model_path", default="bushu/server_model/electra_smallex", type=str, help="模型路径")
    parser.add_argument("--use_GPU", default=True, type=bool, help="使用GPU")
    parser.add_argument("--batch_size", default=32, type=int, help="sents num of one batch")
    args = parser.parse_args()
    load_model(args)
    app.run(host='0.0.0.0')