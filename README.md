# Electra_CRF_NER
1. 模型结构采用：预训练模型+CRF

2. 我们的测试环境为2个Tesla P4 一块显存8GB

3. 使用Flask部署，以调用服务的形式测速，速度为 BERT-base+CRF:4600字/秒 Albert_small+CRF:21000字/秒 Electra_small+CRF:16000字/秒

4. 为了增强训练效果，我们使用了金融新闻作为预训练数据，在开源发布的两个中文模型Albert_small和Electra_small基础之上进行预训练，训练后对下游任务的训练速度和效果都有提升。

5. 效果上，相比于BERT, Electra_small性能和速度较为均衡，Albert_small因为参数共享机制，拟合和泛化能力都显著减弱
