import yaml
import jieba

jieba.dt.cache_file = '/home/fengmingjian/src/src_2/jieba-dict.txt'

from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_community.vectorstores import FAISS
import datetime
import time
import pathlib
import fire
import re
import torch.multiprocessing as mp

from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用第一张GPU


def get_files(path, start=-1, n=-1):
    """
    获取文件夹下文件列表
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


def get_emb_func(model_name, emb_type="openai"):
    if emb_type == "local":
        embeddings_func = HuggingFaceEmbeddings(
            model_name=model_name,
            #model_kwargs={"device": "cuda"},
            model_kwargs={"device": "cuda:0"},
        )
    return embeddings_func


def get_words(text):
    text = text.lower()
    doc_words = jieba.cut(text)
    return list(doc_words)


def get_stop_words(file):
    with open(file, 'r') as r:
        stop_words = r.readlines()
    stop_words = [i[:-1] for i in stop_words]
    return stop_words


def update_word_list(sentence, words):
    sentence = sentence.lower()
    i = 0
    update_words = []
    for i_word, word in enumerate(words):
        if sentence[i:i + len(word)] == word:
            update_words.append(word)
            i += len(word)
        else:
            if word in sentence[i:]:
                index = sentence[i:].index(word)
                update_word = sentence[i:i + index + 1]
                update_words.append(update_word)
                i += index + 1
            else:
                next_i = i_word + 1
                while next_i < len(words) and words[next_i] not in sentence[i:]:
                    next_i += 1
                if next_i == len(words):
                    update_word = sentence[i:]
                    update_words.append(update_word)
                    break
                else:
                    next_word = words[next_i]
                    index = sentence[i:].index(next_word)
                    update_word = sentence[i:i + index + 1]
                    update_words.append(update_word)
                    i += index + 1
    return update_words


def get_short_text_emb(text, embeddings_func, stop_word):
    words = []
    embeddings = []
    doc_words = get_words(text)

    doc_tokens = embeddings_func.client.encode(text, output_value="token_embeddings")[1:-1]
    tokenize_words_ = embeddings_func.client.tokenizer.tokenize(text)
    tokenize_words = []
    for i in tokenize_words_:
        if i.startswith('##'):
            tokenize_words.append(i[2:])
        else:
            tokenize_words.append(i)
    # print("text:", text)
    # print("doc_words:", doc_words)
    # print("before tokenize_words:", tokenize_words)
    tokenize_words = update_word_list(text, tokenize_words)
    # print("update tokenize_words:", tokenize_words)
    start_index = 0
    prefix = 0
    for word in doc_words:
        end_index = start_index + 1
        while word not in ''.join(tokenize_words[start_index:end_index])[prefix:]:
            end_index += 1
        if word not in stop_word:
            emb = torch.mean(doc_tokens[start_index:end_index], dim=0).cpu().numpy()
            words.append(word)
            embeddings.append(emb)
        # print("start_index:", start_index)
        # print("end_index:", end_index)
        # print(tokenize_words[start_index:end_index])
        # print(tokenize_words[start_index:end_index][prefix:])
        if word == ''.join(tokenize_words[start_index:end_index])[prefix:]:
            start_index = end_index
            prefix = 0
        else:
            prefix = len(tokenize_words[end_index - 1]) - (
                    len(''.join(tokenize_words[start_index:end_index])[prefix:]) - len(word))
            start_index = end_index - 1

    # print("words:", words)

    if len(words) == 0:
        # print("len(words) == 0")
        # print(text)
        # input()
        return None, None

    return words, embeddings

# def get_short_text_emb(text, embeddings_func, stop_word):
#     embeddings = embeddings_func.client.encode(text, output_value="token_embeddings")[1:-1]
#     if torch.is_tensor(embeddings):
#         embeddings = [emb.cpu().numpy() for emb in embeddings]  # 将每个embedding移到CPU并转换为numpy
#     words = embeddings_func.client.tokenizer.tokenize(text)
#     return words, embeddings


def get_text_emb(text, embeddings_func, stop_word):
    try:
        words = []
        embeddings = []


        if len(text) >= 510:
            # print(text)
            texts = re.split('(。|！|，|、|？|-|\(|●|l|[1-9]|<|a|/|\.|,|\?)', text)
            i_texts = 0
            while i_texts < len(texts):
                text_ = ''
                start_index = i_texts
                while i_texts < len(texts) and len(text_) + len(texts[i_texts]) < 510:
                    text_ += texts[i_texts]
                    i_texts += 1
                word, embedding = get_short_text_emb(text_, embeddings_func, stop_word)
                if word is not None:
                    words.extend(word)
                    embeddings.extend(embedding)
        else:
            words, embeddings = get_short_text_emb(text, embeddings_func, stop_word)

        if words is None or len(words) == 0:
            return None, None
        return zip(words, embeddings), len(words)
    except Exception as e:
        print(e)



def construct_one_file_vectors(doc_file, embeddings_func, stop_word, process_count=None):
    with open(doc_file, 'r') as f:
        doc = f.read()
        doc = doc.strip()
        doc = doc.lower()
        doc = doc.replace(' ', '')
    try:
        # print(doc_file)
        if len(doc) == 0:
            return None, None
        texts_embeddings, words_count = get_text_emb(doc, embeddings_func, stop_word)
        if texts_embeddings is not None:
            metadata = [{'source': doc_file}] * words_count
        else:
            print(doc_file)
            print(doc)
            print('========')
            metadata = None
        if process_count is not None:
            process_count.value += 1
            print(f"{process_count.value} time:{datetime.datetime.fromtimestamp(time.time())}")
        return texts_embeddings, metadata
    except Exception as e:
        print(e)


def construct_and_save_one_file_vectors(doc_file, embeddings_func, stop_word, vector_path, process_count=None):
    texts_embeddings, metadata = construct_one_file_vectors(doc_file, embeddings_func, stop_word, process_count)
    if texts_embeddings is not None:
        sub_vector = FAISS.from_embeddings(texts_embeddings, embeddings_func, metadata)
        sub_vector_path = vector_path + doc_file.stem
        save_vector(sub_vector, sub_vector_path)
    return 0


def save_vector(vector, vector_path):
    vector.save_local(vector_path)


def check_cache(doc_files, vector_path):
    result_files = []
    for doc_file in doc_files:
        sub_vector_path = vector_path + doc_file.stem
        if not pathlib.Path(sub_vector_path).exists():
            result_files.append(doc_file)
        # else:
        #     print(f"{sub_vector_path} exists")
        #     input()
    return result_files


def construct_vector(doc_files, embeddings_func, vector_path, stop_word_path, start, end, parrallel_num, use_cache):
    stop_word = get_stop_words(stop_word_path)
    doc_files = doc_files[start:end]
    print(len(doc_files))
    if use_cache:
        doc_files = check_cache(doc_files, vector_path)
    print(len(doc_files))
    if parrallel_num == -1:
        for doc_file in tqdm(doc_files, desc='Processing'):
            texts_embeddings, metadata = construct_one_file_vectors(doc_file, embeddings_func, stop_word)
            if texts_embeddings is not None:
                sub_vector = FAISS.from_embeddings(texts_embeddings, embeddings_func, metadata)
                sub_vector_path = vector_path + doc_file.stem
                save_vector(sub_vector, sub_vector_path)
    else:
        mp.set_start_method('spawn', force=True)
        process_count = mp.Manager().Value('i', 0)
        print(f"Start time:{datetime.datetime.fromtimestamp(time.time())}")
        with mp.Pool(processes=parrallel_num) as pool:
            results = [
                pool.apply_async(construct_and_save_one_file_vectors,
                                 (doc_file, embeddings_func, stop_word, vector_path, process_count,)) for
                doc_file in doc_files]
            sub_results = [result.get() for result in results]
        for result in sub_results:
            if result != 0:
                print("result != 0,result is ",result)


def main(start_file=0, end_file=100000, parrallel_num=8, config_file='../configs_2_simple/sentence-4.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    start_file = int(start_file)
    end_file = int(end_file)

    doc_files = get_files(config['init_sentence_doc_path'])
    print(len(doc_files))
    embeddings_func = get_emb_func(config['emb_model_name'], config['emb_type'])

    start = time.time()
    construct_vector(doc_files, embeddings_func, config['init_vector_path'], config['stop_words_path'], start_file,
                     min(end_file, len(doc_files)), parrallel_num, use_cache=True)
    print(time.time() - start)


if __name__ == "__main__":
    # 每个文件存储一个数据库

    fire.Fire(main)
