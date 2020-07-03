# -*- coding:utf-8 -*-
from processors.utils_ner import get_entity_bio
import jieba
import re
from tqdm import tqdm
import fool

def rule_foolnltk(tokens=None,labels=None):
    Havefix=False
    nofixlabels=labels.copy()

    sent = "".join(tokens)
    _, ner = fool.analysis(sent)
    for com in ner[0]:
        if "company" in com:
            s,e=com[:2]
            fool_label=["I-ORG"]*(e-s)
            fool_label[0]="B-ORG"
            if fool_label!= labels[s:e] and labels[s]=="O":# 仅在fool预测出新实体时进行修正
                labels[s:e]=fool_label
                Havefix=True
    # if Havefix:
    #     print("foolnltk修复：")
    #     print(tokens)
    #     print(nofixlabels)
    #     print(labels)
    return Havefix,labels

def rule_company(tokens=None,labels=None,entitys=None,postfixs=["公司", "集团","有限公司","股份公司","有限责任公司","股份有限公司","投资有限公司","集团有限公司","资产投资有限公司","无线责任公司"]):
    """补充有限公司词尾
        args:postfix:从短到长
        tokens = ['华', '为', '公', '司', '近', '期', '收', '够', '了', '小', '米', '公', '司'],
        labels = ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG','O', 'O'],
        id2label = {0: "O", 1: "X", 2: "B-ORG", 3: "I-ORG"},
        postfixs = ["公司", "有限公司", "股份公司", "有限责任公司", "股份有限公司"]
        entitys = get_entity_bio(labels, id2label)
        out:['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORGO', 'O']
    """
    Havefix=False
    nofixlabels=labels.copy()

    sent="".join(tokens)
    for entity in entitys:
        if entity[0]!=-1:
            end=entity[2]+1
            for postfix in postfixs:
                l=len(postfix)
                if l<=len(sent)-end+1 and sent[end:end+l]==postfix:
                    Havefix=True
                    labels[end:end+l]=["I-ORG"]*l
                    break
    if Havefix:
        print("公司后缀修复：")
        print(tokens)
        print(nofixlabels)
        print(labels)
    return Havefix,labels

def rule_jieba(tokens=None,labels=None,entitys=None):
    """ 使用jieba分词边界
        tokens = ['华', '为', '公', '司', '近', '期', '收', '购', '了', '小', '米', '公', '司']
        labels = ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG','O', 'O']
        id2label = {0: "O", 1: "X", 2: "B-ORG", 3: "I-ORG"}
        entitys = get_entity_bio(labels, id2label)

    """
    Havefix=False
    nofixlabels=labels.copy()

    index0=0
    seg_dict={}
    sent = "".join(tokens)
    for cutword in jieba.cut(sent):
        index1 = index0 + len(cutword)
        for i in range(index0,index1):
            seg_dict[i]=index1-1
        index0 = index1
    for entity in entitys:
        end = entity[2]
        if end< seg_dict[end]:
            Havefix=True
            labels[end+1:seg_dict[end]+1]=["I-ORG"]*(seg_dict[end]-end)
    if Havefix:
        print("结巴修复：")
        print(tokens)
        print(nofixlabels)
        print(labels)
    return Havefix,labels

def relu_entity(ORG_set,tokens_labels_pair):
    """
        function:根据数据集中出现的实体库进行补充标注
    """
    for i, (tokens, labels) in enumerate(tqdm(tokens_labels_pair)):
        sent="".join(tokens)
        nofixlabels = labels.copy()
        Havefix=False
        for company in ORG_set:
            # print(company,sent)
            find_res=re.finditer(company,sent)
            for res in find_res:
                s,e = res.span()
                true_label=["I-ORG"]*(e-s)
                true_label[0]="B-ORG"
                if labels[s:e]=="O"*(e-s):
                    labels[s:e]=true_label
                    Havefix = True

        if Havefix:
            print("未标注实体修复：")
            print(tokens)
            print(nofixlabels)
            print(labels)
            Havefix=False
        tokens_labels_pair[i]=(tokens, labels)
    return tokens_labels_pair

def reset(file_path,tokens_labels_pair):
    with open(file_path+".fixd", "a",encoding="utf-8") as f:
        print(f"正在写入文件:{file_path}"+".fixd")
        To_write=""
        num_line=0
        for tokens,labels in tokens_labels_pair:
            for token,label in zip(tokens,labels):
                To_write+=token+" "+label+"\n"
                num_line+=1
                if num_line%10000==0and num_line>0:
                    f.writelines(To_write)
                    To_write = ""
                    print(f"已经写入{num_line}行")
            To_write+="\n"
            num_line+=1
            if num_line % 10000 == 0 and num_line > 0:
                f.writelines(To_write)
                To_write = ""
                print(f"已经写入{num_line}行")
        f.writelines(To_write)


def fix_data(file_path,use_foolnltk=False):
    #统一转成BIO换行，没有别的
    label_list=["O","B-ORG","I-ORG"]
    id2label = {0: "O", 1: "B-ORG", 2: "I-ORG"}
    ORG_set=set()#实体库
    num_sents=0
    with open(file_path,encoding="gbk") as f:
        lines=f.readlines()
        tokens_labels_pair=[]
        tokens=[]# one sent
        labels=[]# one sent
        for line in tqdm(lines):
            if line!="\n" and line!="end\n":
                line2= line.strip().split()
                if len(line2)==2 and line2[1] in label_list:
                    tokens.append(line2[0])
                    if line2[1]=="E-ORG":
                        line2[1]=line2[1].replace("E","I")
                    labels.append(line2[1])

                else:
                    print(f"格式错误,忽略处理:{line}")
            else:#句子结尾
                line="\n"
                entitys = get_entity_bio(labels, id2label)

                ## 使用foolnltk修复
                if use_foolnltk:
                    Havefix2, labels = rule_foolnltk(tokens, labels)
                    if Havefix2:#更新实体
                        entitys = get_entity_bio(labels, id2label)

                ## 按公司后缀修复
                Havefix1,labels=rule_company(tokens,labels,entitys)
                if Havefix1:#更新实体
                    entitys = get_entity_bio(labels, id2label)

                ## 结巴分词边界修复 效果不佳 可以去掉
                # Havefix2, labels = rule_jieba(tokens, labels, entitys)
                # if Havefix2:#更新实体
                #     entitys = get_entity_bio(labels, id2label)

                for entity in entitys:
                    entity_str = "".join(tokens)[entity[1]:entity[2]+1]
                    reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
                    entity_str=re.sub(reg, '', entity_str)
                    ORG_set.add(entity_str)#更新实体库
                tokens_labels_pair.append((tokens,labels))
                tokens = []  # one sent
                labels = []  # one sent
                num_sents+=1
                if num_sents%5000==0:
                    reset(file_path, tokens_labels_pair)
                    tokens_labels_pair=[]
                    print(f"已写入{num_sents}句子")
    ## 补充修复 数据集大时非常慢 暂时去掉
    # tokens_labels_pair=relu_entity(ORG_set, tokens_labels_pair)
    reset(file_path, tokens_labels_pair)

if __name__ == "__main__":
    base_path = "/root/zhanghan/Albert_pretraining/dataset/corpus/trained_data/split_corpus/"
    for fName in ["corpus.txt.O"]:
        file_path=base_path+fName
        fix_data(file_path,use_foolnltk=True)
