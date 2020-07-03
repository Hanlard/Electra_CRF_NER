from processors.utils_ner import get_entity_bio
import jieba
jieba.load_userdict("fix_by_rule/jieba_wholewords.txt")
# jieba.load_userdict("jieba_wholewords.txt")
import collections


def rule_company(tokens=None,labels=None,entitys=None,
                 postfixs=["公司","集团", "有限公司", "股份公司", "有限责任公司", "股份有限公司","投资有限公司","集团有限公司","资产投资有限公司"]):
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
    if Havefix :
        print("公司修复：")
        print(tokens)
        print(nofixlabels)
        print(labels)
    return labels

def rule_jieba(tokens=None,labels=None,entitys=None):
    """ 使用jieba分词边界
        tokens = ['华', '为', '公', '司', '近', '期', '收', '购', '了', '小', '米', '公', '司']
        labels = ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG','O', 'O']
        id2label = {0: "O", 1: "X", 2: "B-ORG", 3: "I-ORG"}
        entitys = get_entity_bio(labels, id2label)

    """
    Havefix=False
    tokens=tokens
    labels=labels

    nofixlabels=labels.copy()

    index0=0
    seg_dict={}
    sent = "".join(tokens)
    sent=sent.replace("[CLS]","CLS")
    sent=sent.replace("[PAD]","PAD")
    sent=sent.replace("[UNK]","UNK")
    sent=sent.replace("[SEP]", "SEP")
    jiebacut=list(jieba.cut(sent))
    for cutword in jiebacut:
        len_cutword=len(cutword)
        if cutword in ["SEP","UNK","PAD","CLS"]:
            len_cutword=1
        index1 = index0 + len_cutword
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
        print(jiebacut)
        print(nofixlabels)
        print(labels)
    return labels

def rule_BIO(labels=None,entitys=None):
    """ 使用jieba分词边界
        tokens = ['华', '为', '公', '司', '近', '期', '收', '购', '了', '小', '米', '公', '司']
        labels = ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG','O', 'O']
        id2label = {0: "O", 1: "X", 2: "B-ORG", 3: "I-ORG"}
        entitys = get_entity_bio(labels, id2label)

    """
    labels=["O"]*len(labels)
    for entity in entitys:
        labels[entity[1]:entity[2]+1]=["I-"+entity[0]]*(entity[2]+1-entity[1])
        labels[entity[1]]="B-"+entity[0]
    return  labels

def fix(input_ids,predict_ids,id2label,tokenizer):
    """
    para:predict -> list
    return: "BIO label" -> list
    """
    label2id={id2label[id] :id for id in id2label}
    tokens=[]
    labels=[]

    for input_id, pre in zip(input_ids, predict_ids):
        tokens.append(tokenizer.ids_to_tokens[input_id])
        labels.append(id2label[pre])
    entitys = get_entity_bio(labels, id2label)

    labels = rule_BIO(labels, entitys)
    labels = rule_jieba(tokens, labels, entitys)# 会对[unk][cls][pad][sep]分割！
    labels =rule_company(tokens,labels,entitys)

    labels_id=[label2id[label] for label in labels]
    return labels_id

if __name__ == "__main__":
    tokens = ['[CLS]', '如', '果', '未', '来', '碧', '桂', '园', '集', '团', '的', '偿', '债', '能', '力', '下', '降', '，', '无', '法', '按', '期', '偿', '还', '上', '述', '所', '述', '的', '债', '务', '本', '息', '，', '债', '权', '人', '依', '法', '履', '行', '股', '权', '质', '押', '合', '同', '，', '则', '公', '司', '将', '存', '在', '实', '际', '控', '制', '人', '变', '更', '风', '险', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
    #['华', '为', '公', '司', '近', '期', '收', '购', '了', '小', '米', '公', '司']
    labels = ['O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    #['B-ORG', 'I-ORG', 'O', 'O', 'O', 'I-OEG', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O']
    id2label = {0: "O", 1: "X", 2: "B-ORG", 3: "I-ORG"}
    entitys = get_entity_bio(labels, id2label)

    # print("修复前：")
    # print(tokens)
    # print(labels)
    # print("BIO修复:")
    # labels = rule_BIO(labels, entitys)
    # print(tokens)
    # print(labels)
    # print("分词修复:")
    labels = rule_jieba(tokens, labels, entitys)
    # print(list(jieba.cut("".join(tokens))))
    # print(tokens)
    # print(labels)
    # print("分词修复+公司后缀修复:")
    # labels = rule_company(tokens, labels, entitys)
    # print(tokens)
    # print(labels)