import json
import time
import yaml
import pathlib
import psutil
import logging
from langchain_community.vectorstores import FAISS

# from construct_token_database_onefile import get_text_emb, get_stop_words
from construct_token_database_onefileenglish import get_text_emb, get_stop_words
import sys
import gc
from tqdm import tqdm
import signal
from contextlib import contextmanager

sys.path.append('../')
from utils import get_file_query, get_emb_func

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='search_process.log'
)

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def monitor_memory():
    """监控内存使用情况"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    #print(f"当前内存使用: {memory_mb:.2f} MB")
    return memory_mb

def get_stage_1_result(file, num):
    """获取第一阶段结果"""
    try:
        result = []
        with open(file, 'r') as f:
            for data in f:
                data = eval(data)
                result.append(data[:num])
        #print(f"成功加载第一阶段结果，共 {len(result)} 条")
        return result
    except Exception as e:
        #print(f"加载第一阶段结果失败: {str(e)}")
        raise

def load_vector(files, vector_path, embeddings_func):
    """加载向量数据"""
    try:
        #print(f"开始加载向量文件...")
        files_name = [i.split('/')[-1].split('.')[0] for i in files]
        vector_path = pathlib.Path(vector_path)
        vector = None
        
        for file in files_name:
            try:
                file_path = vector_path.joinpath(file)
                #print(f"正在加载: {file_path}")
                
                if not file_path.exists():
                    print(f"文件不存在: {file_path}")
                    continue
                    
                sub_vector = FAISS.load_local(str(file_path), embeddings_func, allow_dangerous_deserialization=True)
                if vector is None:
                    vector = sub_vector
                else:
                    vector.merge_from(sub_vector)
                #print(f"成功加载向量文件: {file}")
                
            except Exception as e:
                print(f"加载文件 {file} 时出错: {str(e)}")
                continue
                
        return vector
        
    except Exception as e:
        print(f"load_vector 函数发生错误: {str(e)}")
        raise

def search(embeddings_func, stage_1_result, queries, answers, vector_path, stop_word, k_word_num, k_num):
    """搜索函数"""
    search_result = {}
    total_time = 0
    query_count = 0
    
    # print("\n开始处理查询...")
    # print(f"总查询数量: {len(queries)}")
    # print(f"第一个查询示例: {queries[0]}")
    # print(f"向量路径: {vector_path}")
    
    with open('/home/fengmingjian/src/results/stage2-CUDtoken.json', 'w') as tmp_result_file:
        for idx, (candidate_files, q, a) in enumerate(tqdm(zip(stage_1_result, queries, answers), total=len(queries))):
            try:
                start_time = time.time()
                #print(f"\n处理第 {idx+1} 个查询: {q}")
                
                # 监控内存
                # memory_usage = monitor_memory()
                # if memory_usage > 1000:
                #     print("执行内存清理...")
                #     gc.collect()
                
                # 加载向量
                vector = load_vector(candidate_files, vector_path, embeddings_func)
                if vector is None:
                    print(f"跳过查询 '{q}' - 无法加载向量")
                    continue
                
                #print("处理词向量...")
                q_word_emb, _ = get_text_emb(q, embeddings_func, stop_word)
                word, emb = zip(*q_word_emb)
                #print(f"得到的词: {word}")
                #print(f"词向量数量: {len(emb)}")
                
                q_result = {}
                tmp_result = {}
                token_i_max_result = {}
                
                #print(f"开始处理每个token...")
                for i, emb_ in enumerate(emb):
                    #print(f"处理第 {i+1}/{len(emb)} 个token: {word[i]}")
                    try:
                        with time_limit(300):  # 5分钟超时
                            result = vector.similarity_search_with_score_by_vector(emb_, k=k_word_num)
                            # print(f"token {word[i]} 相似度搜索完成")
                    except TimeoutException:
                        # print(f"处理token {word[i]} 超时，跳过")
                        continue
                    except Exception as e:
                        # print(f"处理token {word[i]} 时出错: {str(e)}")
                        continue
                        
                    _, token_i_max_result[i] = result[-1]
                    
                    page_contents = []
                    scores = []
                    sources = []
                    tmp_result[i] = {
                        "word": word[i],
                        "content": [],
                        "source": [],
                        "score": []
                    }
                    
                    for result_ in result:
                        meta, score = result_
                        page_content = meta.page_content
                        metadata = meta.metadata['source']
                        
                        page_contents.append(page_content)
                        scores.append(str(score))
                        sources.append(metadata.stem)
                        
                        if metadata not in q_result:
                            q_result[metadata] = {}
                        if i not in q_result[metadata]:
                            q_result[metadata][i] = score
                        else:
                            q_result[metadata][i] = min(q_result[metadata][i], score)
                            
                    tmp_result[i]["content"] = page_contents
                    tmp_result[i]["source"] = sources
                    tmp_result[i]["score"] = scores
                
                print("写入临时结果...")
                tmp_result_file.write(json.dumps({q: tmp_result}, ensure_ascii=False) + '\n')
                
                print("计算最终得分...")
                q_result_file = {}
                query_len = len(emb)
                
                for file, token_score in q_result.items():
                    q_result_file[file] = 0
                    for j in range(query_len):
                        if j not in token_score:
                            q_result_file[file] += token_i_max_result[j] + 5
                        else:
                            q_result_file[file] += token_score[j]
                
                q_result_file = list(sorted(q_result_file.items(), key=lambda x: x[1]))
                k_file = q_result_file[:k_num]
                search_result[q] = k_file
                
                process_time = time.time() - start_time
                total_time += process_time/len(emb)
                query_count += 1
                print(f"查询处理完成，耗时: {process_time:.2f}秒")
                
                del vector
                gc.collect()
                
            except Exception as e:
                print(f"处理查询 '{q}' 时发生错误: {str(e)}")
                continue
    
    avg_time = total_time / query_count if query_count > 0 else 0
    print(f"\n平均检索时间: {avg_time:.2f}秒")    
    return search_result

def save_result(results, file, labels=None):
    """保存结果到文件"""
    try:
        print(f"开始保存结果到: {file}")
        with open(file, 'w') as w:
            for i, (q, file_scores) in enumerate(results.items()):
                data = []
                for file, score in file_scores:
                    try:
                        with open(file, 'r') as r:
                            data.append([file.stem, r.read(), score])
                    except Exception as e:
                        print(f"读取文件 {file} 失败: {str(e)}")
                        continue
                
                w.write(json.dumps({q: {"label": labels[i], "result": data}}, ensure_ascii=False) + '\n')
        print("结果保存完成")
    except Exception as e:
        print(f"保存结果时发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("开始执行程序")
        print("加载配置文件...")
        
        config_file = '../configs_2_simple/english.yaml'
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        print("配置加载完成，开始初始化...")
        
        query_type = config['query_type']
        vector_path = config['same_file_vector_path']
        stop_words_path = config['stop_words_path']
        candidate_num = config['candidate_content']
        candidate_word_num = config['candidate_word']
        result_stage_2_file = config['result_stage_2_file']
        
        print(f"向量路径: {vector_path}")
        print(f"停用词路径: {stop_words_path}")
        
        embeddings_func = get_emb_func(config['emb_model_name'], config['emb_type'])
        stop_word = get_stop_words(stop_words_path)
        
        if query_type == 'input':
            print("使用输入模式")
            pass
        else:
            print("使用文件模式")
            stage_1_result_file = config['stage_1_result']
            stage_1_num = config['stage_1_num']
            query_file = config['query_file']
            
            print(f"加载查询文件: {query_file}")
            queries, answers = get_file_query(query_file)
            print(f"查询数量: {len(queries)}")
            
            print(f"加载第一阶段结果: {stage_1_result_file}")
            stage_1_result = get_stage_1_result(stage_1_result_file, stage_1_num)
            print(f"第一阶段结果数量: {len(stage_1_result)}")
            
            
            search_result = search(embeddings_func, stage_1_result, queries, answers, 
                                 vector_path, stop_word, candidate_word_num, candidate_num)
            
            save_result(search_result, result_stage_2_file, answers)
        
        print("程序执行完成")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise