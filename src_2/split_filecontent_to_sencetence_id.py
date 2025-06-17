import yaml
import pathlib
import re
import tqdm

# from split_filecontent_to_sencetenceenglish import split_files_content_to_sentence
# from split_filecontent_to_sencetence import split_files_content_to_sentence


def get_files_stage1(file, file_num):
    results = []
    count = 0
    with open(file, 'r') as f:
        for data in f:
            count+=1
            
            data = eval(data)
            top_data = data[:file_num]
            results.extend(top_data)
            if count < 10:
                print(top_data)
    return set(results)


if __name__ == "__main__":
    config_file = '../configs_2_simple/sentence-4.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # 将文章切分为句子
    origin_doc_path = config['origin_doc_path']
    sentence_doc_path = config['init_sentence_doc_path']
    sentence_num = config['init_sentence_num']
    step_size = config['init_step_size']

    # files = get_files(origin_doc_path)
    file_num = config['stage_1_num']
    stage_1_result = config['stage_1_result']
    doc_files = get_files_stage1(stage_1_result, file_num)
    #print(doc_files)

    results = split_files_content_to_sentence(doc_files, sentence_doc_path, sentence_num, step_size)

