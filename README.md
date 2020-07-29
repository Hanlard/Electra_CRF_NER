# ABSTRACT

Training large deep neural networks needs massive high quality annotation data, but the time and labor costs are too expensive for small business. We start a company-name recognition task with a small scale and low quality training data, then using skills to enhanced model training speed and predicting performance with least artificial participation. The methods we use involve lite pre-training models such as Albert-small or Electra-small with financial corpus, knowledge of distillation and multi-stage learning. The result is that we improve the recall rate of company names recognition task from 0.73 to 0.92 and get 4 times as fast as BERT-Bilstm-CRF model.

# Dataset
url：https://pan.baidu.com/s/1isI-n1hOmP6nq8hO6SJKsw 

password：0klm

# Electra_CRF_NER
1. 模型结构采用：预训练模型+CRF

2. 我们的测试环境为1个Tesla P4 显存8GB

3. 使用Flask部署，以调用服务的形式测速，速度为 BERT-base+CRF:4600字/秒 Albert_small+CRF:21000字/秒 Electra_small+CRF:16000字/秒

4. 为了增强训练效果，我们使用了金融新闻作为预训练数据，在开源发布的两个中文模型Albert_small和Electra_small基础之上进行预训练，训练后对下游任务的训练速度和效果都有提升（5个点召回率提升和8倍收敛速度提升）。

5. 效果上，相比于BERT, Electra_small性能和速度较为均衡，Albert_small因为参数共享机制，拟合和泛化能力都显著减弱

 6.关于数据标注和修复：1）我们使用YEDDA开源标注工具 2）开发了3种数据修复策略，包括分词边界修复，公司后缀修复和Foolnltk工具修复，见datafix.py

 7.关于部署：使用flask部署，有4个版本可以调用：server_{model_name}.py 调用时使用统一接口，将文章列表传入即可

 8.对ALbert词表进行扩充，原始版本的中文vocab.txt缺少中文双引号“”，空格等常见字符，见new_voab

 9.开发了格式转换工具，可以将模型预测和标注不一致的数据进行合并处理，转化为YEDDA标注格式，将模糊实体进行高亮显示，提升人工标注效率

 10.提升数据质量上，我们使用trie树和foolnltk工具对35万文本数据进行预标注，作为训练集，开发集采用高质量人工标注的100篇新闻，达到过拟合之前停止训练，然后预测训练集，找到预测错误的句子（占1/4)，使用9描述的方法校验提升质量

11. 利用知识蒸馏：使用Electra_base+CRF训练的模型对162万篇文章进行标注，标注后数据用于Electra_small+CRF模型的训练，实体召回率从84提升至90（提升5至6个百分点）。
