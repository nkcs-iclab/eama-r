from langchain_community.llms import Ollama, HuggingFacePipeline

from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline, AutoTokenizer, AutoModelForCausalLM, \
    GenerationConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import json
import pathlib
import os
import time
import torch
import pandas as pd
import concurrent.futures
import multiprocessing as mp
import yaml

from utils import get_files, create_all_vector, get_emb_func

os.environ['OPENAI_API_KEY'] = 'sk-hJwYsJk4FdTE0Wbab0ruOIEuJddhjfUoczsALaHy9UnzbCIL'
os.environ['OPENAI_API_BASE'] = 'https://api.chatanywhere.tech/v1'


def create_llm(model_path, model_type="ollama"):
    if model_type == "ollama":
        llm = Ollama(model=model_path)
    elif model_type == "openai":
        llm = ChatOpenAI(model_name=model_path, temperature=0)
    elif model_type == "local":
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        base_model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            device_map='auto',
        )
        pipe = pipeline(
            "text-generation",
            model=base_model,
            tokenizer=tokenizer,
            max_length=2048,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.2
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    elif model_type == "local_auto":
        model_name = config['model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        max_memory = {i: "32GB" for i in range(3)}
        base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="sequential",
                                                          torch_dtype=torch.bfloat16, max_memory=max_memory,
                                                          attn_implementation="eager")
        base_model.generation_config = GenerationConfig.from_pretrained(model_name)
        base_model.generation_config.pad_token_id = base_model.generation_config.eos_token_id
        # input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        # outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)
        #
        # results = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        pipe = pipeline(
            "text-generation",
            model=base_model,
            tokenizer=tokenizer,
            max_length=2048,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.2
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def input_rag():
    while True:
        query = input("input your query:")
        # query ='无纺布全包覆卫生棉条如何使用'
        candidate_split_hyper = [eval(i) for i in config['candidate_split_hyper']]
        (query, ground_truth_answer, before_rag_context, before_rag_answer, after_rag_context,
         after_rag_answer) = query_one(query,
                                       vectors,
                                       config['candidate_docs_num'],
                                       candidate_split_hyper,
                                       config['candidate_content'],
                                       vector_path,
                                       embeddings_func,
                                       config['candidate_docs_dir'],
                                       config['model_name'],
                                       config['model_type'],
                                       debug=config['debug'])
        print("query:", query)
        print('====================')
        print("ground_truth_answer:", ground_truth_answer)
        print('====================')
        print("before_rag_context:", before_rag_context)
        print('====================')
        print("before_rag_answer:", before_rag_answer)
        print('====================')
        print("after_rag_context:", after_rag_context)
        print('====================')
        print("after_rag_answer:", after_rag_answer)
        print('====================')
    #     query = '机器人可以完成哪些功能'
    #     query = '带有液压升降的可折叠羽毛球网架解决了什么问题？'
    #     query = '电化学装置的隔板通常使用什么材料制成？'
    #     query = '电化学装置隔板使用的多孔涂层的材料有哪些？'


def query_one(query, vectors, candidate_docs_num, candidate_split_hyper, candidate_content, vector_path,
              embeddings_func, candidate_docs_dir, model_name, model_type, debug=False, ground_truth_answer=None):
    llm = create_llm(model_name, model_type=model_type)
    result = vectors.similarity_search_with_score(query, k=candidate_docs_num)
    candidate_docs_dir = pathlib.Path(candidate_docs_dir)
    candidate_docs = set()
    for i in result:
        meta = i[0].metadata
        for k, v in meta.items():
            candidate_docs.add(candidate_docs_dir.joinpath(v.name))
        if debug:
            print("init results:", i[0].page_content)
            print("meta patent-data:", i[0].metadata)
            print("score:", i[1])
            print('=================== ')
            input("continue?")

    if debug:
        print("candidate_docs:", len(candidate_docs))
    if debug:
        for doc in candidate_docs:
            with open(doc, 'r') as f:
                print(f.readlines()[0][:300])
            print('--------------------------')
        input("continue?")

    multiquerys = llm(f"""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines.
        Original question: {query}""")
    fake_answer = llm(f"""You are an AI language model assistant. Your task is to answer the following question.
        Original question: {query}""")
    query = [query] + multiquerys.split('\n')
    print("fake_answer:", fake_answer)
    candidate_split_hyper = [(len(fake_answer), 0, ['。'], 10)]
    candidate_docs_vectors = create_all_vector(candidate_split_hyper, vector_path, candidate_docs, embeddings_func,
                                               use_save=True, save=True, save_target='file')
    results = []
    for q in query:
        sub_result = candidate_docs_vectors.similarity_search_with_score(q, k=candidate_content)
        print("q:", q)
        print("sub_result:", ('\n').join([i[0].page_content for i in sub_result]))
        print('=================== ')
        results.extend(sub_result)
    before_rag_context = []
    after_rag_context = []
    # if debug:
    #     print('results:')
    contents = set()
    result = []
    for i in results:
        if i[0].page_content not in contents:
            result.append(i)
            contents.add(i[0].page_content)
    for i in result:
        prompt = f"判断参考内容是否能解决问题，请使用“可以”或者“不可以”回答问题。\n 参考内容：{i[0].page_content} \n 问题：{query[0]}"
        yesorno = llm(prompt)
        if yesorno.startswith('可以'):
            after_rag_context.append(i)
        before_rag_context.append(i)
    after_prompt = f"请根据参考内容回答问题。\n 参考内容：{after_rag_context}\n问题：{query[0]}。如果不能回答问题则直接说无法回答。"
    after_rag_answer = llm(after_prompt)
    before_prompt = f"请根据参考内容回答问题。\n 参考内容：{before_rag_context}\n问题：{query[0]}。如果不能回答问题则直接说无法回答。"
    before_rag_answer = llm(before_prompt)
    return (query, ground_truth_answer, before_rag_context, before_rag_answer, after_rag_context, after_rag_answer)


def file_rag(config, vectors, vector_path, embeddings_func, queryies, answers,
             before_result_file='../results/before.json',
             after_result_file='../results/after.json'):
    mp.set_start_method('spawn', force=True)
    start = time.time()
    num = 3
    # queryies = queryies[:num]
    # answers = answers[:num]
    before_result_file = open(before_result_file, 'w')
    after_result_file = open(after_result_file, 'w')
    candidate_split_hyper = [eval(i) for i in config['candidate_split_hyper']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num) as executor:

        futures = [
            executor.submit(query_one, i_q, vectors, config['candidate_docs_num'], candidate_split_hyper,
                            config['candidate_content'],
                            vector_path,
                            embeddings_func,
                            config['candidate_docs_dir'],
                            config['model_name'],
                            config['model_type'],
                            config['debug'], i_a) for
            (i_q, i_a) in
            zip(queryies, answers)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            before_rag_context = []
            after_rag_context = []

            for context in result[2]:
                before_rag_context.append(context.page_content)
            for context in result[4]:
                after_rag_context.append(context.page_content)
            before_result_file.write(
                json.dumps({'query': result[0], 'answer': result[1], 'rag_context': before_rag_context,
                            'rag_result': result[3]}, ensure_ascii=False) + '\n', )
            after_result_file.write(
                json.dumps({'query': result[0], 'answer': result[1], 'rag_context': after_rag_context,
                            'rag_result': result[5]}, ensure_ascii=False) + '\n')

    before_result_file.close()
    after_result_file.close()
    end = time.time()
    print("time:", end - start)


if __name__ == "__main__":

    # 从local.yaml中读取超参数
    config_file = 'configs_1_simple/local-2.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    split_hyper = [eval(i) for i in config['split_hyper']]
    vector_path = config['vector_path'] + config['emb_model_name'].split("/")[-1]
    doc_files = get_files(config['doc_path'])

    embeddings_func = get_emb_func(config['emb_model_name'], config['emb_type'])
    vectors = create_all_vector(split_hyper, vector_path, doc_files, embeddings_func,
                                config['cluster_method'],
                                config['cluster_hyperparams'], use_save=True, save=True, save_target='dir')

    if config['query_type'] == "file":
        #     读取xlsx文件
        queryies, answers = get_file_query(config['query_file'])
        file_rag(config, vectors, vector_path, embeddings_func, queryies, answers)
    else:
        input_rag()
