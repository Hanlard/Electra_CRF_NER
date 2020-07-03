import foolnltk as fool
import re

context_text="华为是中国的科技公司"
words, ner = fool.analysis(context_text)
coms = []
print('ner',ner)
for com in ner[0]:
    # print('com',com)
    if 'company' in com:
        ner_company = re.sub(r'[\n\r\s\t|\'\"\u3000]','',com[-1])
        # ner_company = com[-1].strip(' ').strip("'").strip('丨').strip('\n').strip(' ').strip('\u3000').strip('\t').strip('\s')
        # print(ner_company)
        coms.append(ner_company)
print(coms)