import yaml
import fire
import pathlib
import time
from construct_token_database_onefileenglish import get_emb_func
from langchain_community.vectorstores import FAISS


def get_files(path):
    path = pathlib.Path(path)
    files = list(path.glob('*'))
    results = {}
    for file in files:
        id = file.stem.split('_')[0]
        if id not in results:
            results[id] = [file]
        else:
            results[id].append(file)
    return results


def merge(vectors_files, embeddings_func, same_file_vector_path,use_cache):

    same_file_vector_path = pathlib.Path(same_file_vector_path)
    same_file_vector_path.mkdir(parents=True, exist_ok=True)
    count = 0
    all_count = len(vectors_files)
    for k, v in vectors_files.items():
        save_path = same_file_vector_path.joinpath(k)
        if use_cache and pathlib.Path(save_path).exists():
            continue
        merge_vector = None
        for file in v:
            sub_vector = FAISS.load_local(file, embeddings_func, allow_dangerous_deserialization=True)
            if merge_vector is None:
                merge_vector = sub_vector
            else:
                merge_vector.merge_from(sub_vector)
        merge_vector.save_local(save_path)
        count += 1
        print(f"{count}/{all_count}")


def main(config_file='../configs_2_simple/english.yaml',use_cache=False):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    merge_vector_path = config['merge_vector_path']
    print(merge_vector_path)
    same_file_vector_path = config['same_file_vector_path']
    vectors_files = get_files(merge_vector_path)

    all_count = len(vectors_files)
    print(all_count)

    embeddings_func = get_emb_func(config['emb_model_name'], config['emb_type'])

    start = time.time()
    merge(vectors_files, embeddings_func, same_file_vector_path,use_cache)
    print(time.time() - start)


if __name__ == "__main__":

    # 每个文件存储一个数据库

    fire.Fire(main)
