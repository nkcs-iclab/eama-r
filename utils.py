from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import psutil
import os
import pathlib
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from clustering import kmeans, agglomerative


def get_files(path, start=-1, n=-1):
    """
    获取文件夹下文件列表
    :param path: 文件路径
    :param n: 文件数量
    :return: 文件列表
    """
    path = pathlib.Path(path)
    files = list(path.glob('*'))
    if start != -1:
        files = sorted(files, key=lambda x: int(x.name.split('.')[0]))
        files = files[start:]
    if n != -1:
        files = sorted(files, key=lambda x: int(x.name.split('.')[0]))
        files = files[:n]
    return files


def get_documents(doc_path, chunk_size, overlap_size, separators, min_chunk_size=10):
    documents = []
    for i, doc_file in enumerate(doc_path):
        if i % 10000 == 0:
            print(f'processing {i}th file')
        loader = TextLoader(doc_file)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=overlap_size,
                                                       length_function=len,
                                                       is_separator_regex=True,
                                                       separators=separators)
        document = text_splitter.split_documents(docs)
        document = [chunk for chunk in document if len(chunk.page_content) >= min_chunk_size]
        documents.append(document)
    return documents


def get_file_query(file):
    format = file.split('.')[-1]
    format = format.lower()
    # queryies = []
    # answers = []
    if format == 'xlsx':
        df = pd.read_excel(file)
        queryies = df['query'].tolist()
        answers = df['answer'].tolist()
        # for i in range(len(df)):
        #     queryies.append(df.iloc[i, 0])
        #     answers.append(df.iloc[i, 1])
    elif format == 'csv':
        df = pd.read_csv(file)
        queryies = df['question'].tolist()
        answers = df['answer'].tolist()
    return queryies, answers

#这个是默认的
# def get_emb_func(model_name, emb_type="openai"):
#     if emb_type == "openai":
#         embeddings_func = OpenAIEmbeddings()
#     elif emb_type == "ollama":
#         embeddings_func = OllamaEmbeddings(model=model_name)
#     elif emb_type == "local":
#         embeddings_func = HuggingFaceEmbeddings(
#             model_name=model_name,
#             model_kwargs={"device": "cuda"},
#             encode_kwargs={"convert_to_tensor": True}
#         )
#     return embeddings_func

#这个是NV模型的
# def get_emb_func(model_name, emb_type="openai"):
#     if emb_type == "openai":
#         embeddings_func = OpenAIEmbeddings()
#     elif emb_type == "ollama":
#         embeddings_func = OllamaEmbeddings(model=model_name)
#     elif emb_type == "local":
#         # 设置模型参数
#         model_kwargs = {
#             "device": "cuda",
#             "trust_remote_code": True  # 添加这个参数到 model_kwargs
#         }
#         encode_kwargs = {
#             "convert_to_tensor": True,
#             "normalize_embeddings": True
#         }
        
#         embeddings_func = HuggingFaceEmbeddings(
#             model_name=model_name,
#             model_kwargs=model_kwargs,
#             encode_kwargs=encode_kwargs
#         )
#     return embeddings_func

def get_emb_func(model_name, model_type):
    """获取embedding函数"""
    if model_type == "huggingface":
        try:
            if "voyage" in model_name.lower():
                # 添加必要的导入
                from typing import List
                import voyageai
                from langchain_core.embeddings import Embeddings
                
                class VoyageAIEmbeddings(Embeddings):
                    def __init__(self, model_name: str):
                        self.client = voyageai.Client()  # 会自动使用环境变量 VOYAGE_API_KEY
                        self.model_name = model_name.split('/')[-1]  # 从 'voyageai/voyage-lite-02-instruct' 提取模型名
                    
                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        """Embed a list of documents using Voyage API"""
                        try:
                            result = self.client.embed(
                                texts, 
                                model=self.model_name,
                                input_type="document"
                            )
                            return result.embeddings
                        except Exception as e:
                            print(f"Voyage API 嵌入错误: {str(e)}")
                            raise
                    
                    def embed_query(self, text: str) -> List[float]:
                        """Embed a single query"""
                        try:
                            result = self.client.embed(
                                [text], 
                                model=self.model_name,
                                input_type="query"
                            )
                            return result.embeddings[0]
                        except Exception as e:
                            print(f"Voyage API 查询嵌入错误: {str(e)}")
                            raise
                
                # 创建 Voyage embeddings 实例
                embeddings_func = VoyageAIEmbeddings(model_name)
                
            else:
                # 其他模型使用标准 HuggingFaceEmbeddings
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings_func = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cuda'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            return embeddings_func
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise
            
    elif model_type == "openai":
        from langchain_community.embeddings import OpenAIEmbeddings
        embeddings_func = OpenAIEmbeddings()
        
    elif model_type == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        embeddings_func = OllamaEmbeddings(model=model_name)
        
    elif model_type == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings_func = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"convert_to_tensor": True}
        )
        
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
        
    return embeddings_func




def format_single_data(ncluster, clusters, metadatas, texts):
    new_metadats = []
    new_texts = []
    for i in range(ncluster):
        i_index = np.where(clusters == i)
        i_meta = {}
        for pos, i in enumerate(i_index[0]):
            i_meta[f'source_{pos}'] = metadatas[i]['source']
        i_text_ = [texts[i] for i in i_index[0]]
        i_text = '##'.join(i_text_)
        # if len(i_text_):
        #     print(i_text)
        #     print('------------')
        new_metadats.append(i_meta)
        new_texts.append(i_text)
    # print('============')
    return new_metadats, new_texts


def format_data(ncluster, clusters, metadatas, texts, centroids):
    new_metadats = []
    new_texts = []
    # print(ncluster)
    # print('------------')
    for i in range(ncluster):
        i_index = np.where(clusters == i)
        i_meta = {}
        pos = 0
        for i in i_index[0]:
            for _, v in metadatas[i].items():
                i_meta[f'source_{pos}'] = v
                pos += 1
        i_text = [texts[i] for i in i_index[0]]
        i_text = '##'.join(i_text)
        new_metadats.append(i_meta)
        new_texts.append(i_text)

    text_embeddings = zip(new_texts, centroids)
    return text_embeddings, new_metadats


def get_one_document(doc_path, chunk_size, overlap_size, separators, min_chunk_size=10):
    loader = TextLoader(doc_path)
    docs = loader.load()
    content = docs[0].page_content
    title = content.split('。')[0].split('：')[1]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=overlap_size,
                                                   length_function=len,
                                                   is_separator_regex=True,
                                                   separators=separators)
    document_ = text_splitter.split_documents(docs)
    document_ = [chunk for chunk in document_ if len(chunk.page_content) >= min_chunk_size]
    document = document_
    document = []
    for d in document_:

        if d.page_content.startswith("。"):
            d.page_content = d.page_content[1:]
        d.page_content = f"以下段落来自：{title}。" + d.page_content
        document.append(d)

    return document

# def get_documents(doc_path, chunk_size, overlap_size, separators, min_chunk_size=10):
    
#     documents = []
#     for i, doc_file in enumerate(doc_path):
#         if i % 10000 == 0:
#             print(f'processing {i}th file')
            
#         # 加载文档
#         loader = TextLoader(doc_file)
#         docs = loader.load()
        
#         # 提取标题
#         content = docs[0].page_content
#         try:
#             title = content.split('。')[0].split('：')[1]
#         except IndexError:
#             print(f"Warning: Could not extract title from file {doc_file}, using filename as title")
#             title = os.path.basename(doc_file)
            
#         # 分割文档
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=overlap_size,
#             length_function=len,
#             is_separator_regex=True,
#             separators=separators
#         )
#         document_ = text_splitter.split_documents(docs)
#         document_ = [chunk for chunk in document_ if len(chunk.page_content) >= min_chunk_size]
        
#         # 处理每个文档片段，添加标题
#         document = []
#         for d in document_:
#             # 删除开头的句号（如果有）
#             if d.page_content.startswith("。"):
#                 d.page_content = d.page_content[1:]
#             # 添加标题前缀
#             d.page_content = f"以下段落来自：{title}" + d.page_content
#             document.append(d)
            
#         documents.append(document)
        
#     return documents


def get_single_file_vector(chunk_size, overlap_size, sperator, min_chunk_size, vector_path, doc_path, embeddings_func,
                           use_save=False,
                           save=False):
    vector = None
    for doc_file in doc_path:
        vector_path_ = f'{vector_path}_chunk{chunk_size}_overlap{overlap_size}_min_chunk_size_{min_chunk_size}_file{doc_file.name}'
        if use_save and pathlib.Path(vector_path_).exists():
            sub_vector = FAISS.load_local(vector_path_, embeddings_func, allow_dangerous_deserialization=True)
        else:
            document = get_one_document(doc_file, chunk_size, overlap_size, sperator, min_chunk_size)
            sub_vector = FAISS.from_documents(document, embeddings_func)
            if save:
                sub_vector.save_local(vector_path_)
        if vector is None:
            vector = sub_vector
        else:
            vector.merge_from(sub_vector)
    return vector


def get_all_file_vector(chunk_size, overlap_size, sperator, min_chunk_size, vector_path, doc_path, embeddings_func,
                        cluster_method,
                        cluster_hyperparams, use_save=False, save=False):
    all_emb_num = 0
    ncluster = 0
    if cluster_method is not None:
        vector_path_ = f'{vector_path}_chunk{chunk_size}_overlap{overlap_size}_min_chunk_size_{min_chunk_size}_cluster_{cluster_method}_hyper{cluster_hyperparams}_min'
    else:
        vector_path_ = f'{vector_path}_chunk{chunk_size}_overlap{overlap_size}_min_chunk_size_{min_chunk_size}__allfile'
    if use_save and pathlib.Path(vector_path_).exists():
        vector = FAISS.load_local(vector_path_, embeddings_func, allow_dangerous_deserialization=True)
    else:
        documents = get_documents(doc_path, chunk_size, overlap_size, sperator, min_chunk_size)
        # print("get documents done")
        # print(f"当前内存使用: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024} GB")
        text_embeddings = []
        new_metadats = []
        new_texts = []
        for i in tqdm(range(len(documents))):
            document = documents[i]
            start = time.time()
            text = [d.page_content for d in document]
            metadata = [d.metadata for d in document]
            if len(text) == 0:
                continue
            embedding = embeddings_func.embed_documents(text)
            all_emb_num += len(embedding)
            # print(f"get {i} embeddings done")
            if len(text) != 1 and cluster_method is not None:
                if cluster_method == 'kmeans':
                    centroids_, clusters_, ncluster_ = kmeans(cluster_hyperparams, embedding)
                elif cluster_method == 'agglomerative':
                    centroids_, clusters_, ncluster_ = agglomerative(cluster_hyperparams, embedding)
                # print(f"get {i} clusters done")
                # print(f"当前内存使用: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024} GB")
                new_metadata, new_text = format_single_data(ncluster_, clusters_, metadata, text)
                text_embeddings.extend(centroids_)
                new_metadats.extend(new_metadata)
                new_texts.extend(new_text)
            else:
                text_embeddings.extend(embedding)
                new_metadats.extend(metadata)
                new_texts.extend(text)
            # print(f"{i} time :{time.time() - start}")
        text_embeddings = zip(new_texts, text_embeddings)
        vector = FAISS.from_embeddings(text_embeddings, embeddings_func, new_metadats)
        if save:
            vector.save_local(vector_path_)
    print(f"all emb num: {all_emb_num}")
    print(f"cluster num: {ncluster}")
    return vector


def create_all_vector(split_hyper, vector_path, doc_path, embeddings_func, cluster_method=None,
                      cluster_hyperparams=None, use_save=False, save=False, save_target='dir'):
    vecotrs = None
    for s_hyper in split_hyper:
        chunk_size, overlap_size, seperator, min_chunk_size = s_hyper
        if save_target == 'dir':
            sub_vector = get_all_file_vector(chunk_size, overlap_size, seperator, min_chunk_size, vector_path, doc_path,
                                             embeddings_func,
                                             cluster_method,
                                             cluster_hyperparams,
                                             use_save=use_save, save=save)
        else:
            sub_vector = get_single_file_vector(chunk_size, overlap_size, seperator, min_chunk_size, vector_path,
                                                doc_path,
                                                embeddings_func,
                                                use_save=use_save, save=save)

        if vecotrs is None:
            vecotrs = sub_vector
        else:
            vecotrs.merge_from(sub_vector)
    return vecotrs
