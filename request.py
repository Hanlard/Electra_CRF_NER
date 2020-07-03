
import requests
import pandas as pd
import numpy as np
from time import time
import argparse
def prepare_data(path="../未标注测试数据/结果输出-原始数据.xlsx",num=10):
    Df = pd.read_excel(path, index_col=0)
    Df["albert_crf"]=""
    paragraphs=Df['content'].to_list()[2000:2000+num]
    mean_tokens=np.mean([len(paragraph) for paragraph in paragraphs])
    print(f"平均文章字数:{mean_tokens}")
    return paragraphs

def predict_result(args,PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'):
    # Initialize image path
    paragraphs=prepare_data(args.path,args.Num_articles)
    payload = {'data': paragraphs}
    # Submit the request.
    start=time()
    r = requests.post(PyTorch_REST_API_URL, data=payload).json()
    # Ensure the request was successful.
    end=time()
    if r['success']:
        for (i, result) in enumerate(r['predictions']):
            print(f"Article No.{i}-entitys:{result}")
        print(f"消耗时间:{end-start}s")
    else:
        print('Request failed')

def predict_articles(articles=["华为是一家中国的高科技公司","小米是华为的友商"],PyTorch_REST_API_URL = 'http://192.168.28.31:5000/predict'):
    # Initialize image path
    payload = {'data': articles}
    # Submit the request.
    import requests
    r = requests.post(PyTorch_REST_API_URL, data=payload).json()
    if r['success']:
        ners=[]
        # print(r['predictions'])
        for (i, result) in enumerate(r['predictions']):# {'华为': [3, 5], '小米': [0, 2]}
            ner=[(result[company][0],result[company][1],'company',company) for company in result]
            ners.append(ner)
        return ners
    else:
        return [[]]*len(articles)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", default=["君正集团近日发布公告称，华泰保险集团已收到银保监会《关于华泰保险集团股份有限公司变更股东的批复》，同意君正集团、内蒙古君正化工有限责任公司分别将持有的华泰保险集团4.72亿股股份、1.43亿股股份转让给安达天平再保险有限公司（下称安达天平），转让完成后，安达天平对华泰保险集团的持股比例将达到25.0823%","由软银集团(SoftBank Group Corp.)支持的保险初创公司Lemonade周四表示，该公司计划通过在美国的首次公开募股(IPO)筹集高达2.86亿美元的资金。","纳斯达克总法律顾问办公室已通知公司，公司股票将于2020年6月29日开盘时停牌，纳斯达克将在所有上诉期限届满后提交退市通知。事实上，在近一个月中，瑞幸咖啡已经收到纳斯达克两次退市通知，目前瑞幸咖啡已经放弃了自我挣扎 。"]*1500)
    args = parser.parse_args()
    ners = predict_articles(args.articles,PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict')
    for i,ner in enumerate(ners):
        print(f"【No.{i}】",ner)