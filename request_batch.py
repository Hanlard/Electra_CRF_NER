
import requests
import pandas as pd
import numpy as np
from time import time
import argparse
def prepare_data(path="../未标注测试数据/结果输出-原始数据.xlsx",num=10):
    Df = pd.read_excel(path, index_col=0)
    Df["albert_crf"]=""
    paragraphs=Df['content'].to_list()
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

def predict_Excel(args,PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'):
    Df = pd.read_excel(args.path, index_col=0)

    paragraphs=Df['正文'].to_list()
    mean_tokens=np.mean([len(str(paragraph)) for paragraph in paragraphs])
    print(f"平均文章字数:{mean_tokens}")
    payload = {'data': paragraphs}

    # Submit the request.
    start=time()
    r = requests.post(PyTorch_REST_API_URL, data=payload).json()
    # Ensure the request was successful.
    end=time()
    print(f"消耗时间:{end - start}s")
    if r['success']:
        Df["Electra_crf"] = r['predictions']
        Df.to_excel(args.path.replace(".xlsx","_predict.xlsx"))
    else:
        print('Request failed')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--Num_articles", default=10, type=int,
                        help="测试文章的数量")
    parser.add_argument("--path", default="datasets/excel_data/Data.xlsx", type=str,
                        help="测试文档的路径")
    args = parser.parse_args()
    # predict_result(args,PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict')
    predict_Excel(args, PyTorch_REST_API_URL='http://127.0.0.1:5000/predict')