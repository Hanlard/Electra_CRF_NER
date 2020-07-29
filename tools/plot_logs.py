import pandas as pd
import re
import matplotlib.pyplot as plt
# filepath="D:\\NLP Tasks\\NER\Electra_CRF_NER\outputs\electra\electra-cner-2020-07-27-03_43_07.log"
filepath="D:\\NLP Tasks\\NER\Electra_CRF_NER\outputs\electra\electra-cner-2020-07-28-07_53_51.log"

with open(filepath,encoding="utf-8") as f:
    doc=f.readlines()
record = {'acc': [], 'recall': [], 'f1': [], 'loss': [], 'origin':[], 'found':[], 'right':[] }
for line in doc:
    if " - loss: " in line and " - origin: " in line:
        line=line.replace("f1","")
        line=re.sub("[a-z:-]","",line)
        line_num=line.split()
        record['acc'].append(float(line_num[0]))
        record['recall'].append(float(line_num[1]))
        record['f1'].append(float(line_num[2]))
        record['loss'].append(float(line_num[3]))
        record['origin'].append(float(line_num[4]))
        record['found'].append(float(line_num[5]))
        record['right'].append(float(line_num[6]))
record=pd.DataFrame(record)

eval_setps=50
step=(record.index*eval_setps).to_list()
# Loss=record["loss"].to_list()
# P=record["acc"].to_list()
# R=record["recall"].to_list()
# F1=record["f1"].to_list()
# origin=record["origin"].to_list()
# found=record["found"].to_list()
# right=record["right"].to_list()

plt.subplot(3,1,1)
plt.plot(step,record.iloc[:,:3])
plt.plot(step,[0.9]*len(step),"k--")
plt.grid()
plt.ylim([0.5,1])
plt.legend(["P","R","F1","0.9"])

plt.subplot(3,1,2)
flood_level=1.5
plt.plot(step,record.iloc[:,3])
plt.plot(step,[flood_level]*len(step),"k--")
plt.grid()
plt.ylim([0,10])
plt.legend(["Loss","Flood Level"])

plt.subplot(3,1,3)
plt.plot(step,record.iloc[:,4:])
plt.grid()
plt.ylim([0,5000])
plt.legend(["origin","found","right"])

plt.show()