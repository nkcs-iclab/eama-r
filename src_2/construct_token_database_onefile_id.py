import yaml
import jieba

jieba.dt.cache_file = '/home/lidongwen/lidongwen/langchain-llama/jieba-dict.txt'

from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_community.vectorstores import FAISS
import datetime
import time
import pathlib
import fire
import torch.multiprocessing as mp

from tqdm import tqdm
from construct_token_database_onefile import get_emb_func, construct_vector


def get_id_path(file_path):
    id_file = {}
    file_path = pathlib.Path(file_path)
    files = list(file_path.glob('*'))
    for file in files:
        id = file.name.split('_')[0]
        if id not in id_file:
            id_file[id] = [file]
        else:
            id_file[id].append(file)
    return id_file


def get_files(path, start=-1, n=-1):
    """
    获取文件夹下文件列表F
    :param path: 文件路径
    :param n: 文件数量
    :return: 文件列表
    """
    path = pathlib.Path(path)
    files = list(path.glob('*'))
    files = sorted(files, key=lambda x: int(x.name.split('.')[0]))

    if start != -1:
        files = files[start:]
    if n != -1:
        files = files[:n]
    return files


def get_id_by_path(file_list_path, file_num):
    result_id = []
    with open(file_list_path, 'r') as f:
        for data in f:
            data = eval(data)
            top_data = data[:file_num]
            for i in top_data:
                result_id.append(i.split('/')[-1].split('.')[0])
    return result_id


def get_files_by_id(doc_files_id, id_path):
    doc_files = []
    for id in doc_files_id:
        doc_files.extend(id_path[id])
    doc_files = sorted(doc_files)
    return doc_files


def main(parrallel_num=6, config_file='../configs_2_simple/sentence-4.yaml'):
    # 根据文章id构造token嵌入向量库
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    stage_1_result = config['stage_1_result']
    file_num = config['stage_1_num']
    doc_files_id = get_id_by_path(stage_1_result, file_num)
    id_path = get_id_path(config['sentence_doc_path'])
    doc_files = get_files_by_id(doc_files_id, id_path)
    start_file = 0
    end_file = len(doc_files)

    embeddings_func = get_emb_func(config['emb_model_name'], config['emb_type'])

    start = time.time()
    print(len(doc_files))
    construct_vector(doc_files, embeddings_func, config['init_vector_path'], config['stop_words_path'], start_file,
                     min(end_file, len(doc_files)), parrallel_num, use_cache=True)
    print(time.time() - start)


if __name__ == "__main__":
    # 每个文件存储一个数据库
    # 指定id读取文件进行构建

    fire.Fire(main)
