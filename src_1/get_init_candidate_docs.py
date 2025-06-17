import os
import yaml
import time
import concurrent.futures
import multiprocessing as mp

os.environ['OPENAI_API_KEY'] = 'sk-hJwYsJk4FdTE0Wbab0ruOIEuJddhjfUoczsALaHy9UnzbCIL'
os.environ['OPENAI_API_BASE'] = 'https://api.chatanywhere.tech/v1'

# 设置只使用第 0 块 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append('../')
from utils import get_emb_func, get_file_query, get_files, create_all_vector


def file_rag(config, vectors, queryies, answers, candidate_docs='/home/fengmingjian/src/results/简单问题-stage1.json', num=-1):
    mp.set_start_method('spawn', force=True)
    start = time.time()

    candidate_docs_file = open(candidate_docs, 'w')
    if num == -1:
        for i_q in queryies:
            candidate_docs_name, _ = query_one(i_q, vectors, config['candidate_docs_num'])
            candidate_docs_file.write(str(candidate_docs_name) + '\n')
    else:

        with concurrent.futures.ProcessPoolExecutor(max_workers=num) as executor:
            futures = [
                executor.submit(query_one, i_q, vectors, config['candidate_docs_num']) for
                i_q in
                queryies]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                candidate_docs_name, _ = result
                candidate_docs_file.write(str(candidate_docs_name) + '\n')

    end = time.time()
    print("time:", end - start)


def input_rag(config, vectors):
    while True:
        query = input("Please input your query:")
        if query == '':
            query = '口腔放疗后患者如何进行张口锻炼，同时防止灰尘和细菌进入口腔？'
        candidate_docs_name, result = query_one(query, vectors, config['candidate_docs_num'])
        # for i in range(len(result)):
        #     print(i,result[i])



def query_one(query, vectors, candidate_docs_num):
    result = vectors.similarity_search_with_score(query, k=candidate_docs_num)
    # print(result)
    candidate_docs_name = []
    for i in result:
        meta = i[0].metadata
        for k, v in meta.items():
            if v not in candidate_docs_name:
                candidate_docs_name.append(v)
    return candidate_docs_name, result


if __name__ == "__main__":
    config_file = '../configs_1_simple/local-5.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # cluster_single_file_emb.query_one= query_one
    embeddings_func = get_emb_func(config['emb_model_name'], config['emb_type'])

    split_hyper = [eval(i) for i in config['split_hyper']]
    vector_path = config['vector_path'] + config['emb_model_name'].split("/")[-1]
    doc_files = get_files(config['doc_path'])

    embeddings_func = get_emb_func(config['emb_model_name'], config['emb_type'])
    vectors = create_all_vector(split_hyper, vector_path, doc_files, embeddings_func,
                                config['cluster_method'],
                                config['cluster_hyperparams'], use_save=True, save=True, save_target='dir')

    if config['query_type'] == "file":
        stage_1_result = config['stage_1_result']
        queryies, answers = get_file_query(config['query_file'])
        file_rag(config, vectors, queryies, answers,candidate_docs=stage_1_result)
    else:
        input_rag(config, vectors)
