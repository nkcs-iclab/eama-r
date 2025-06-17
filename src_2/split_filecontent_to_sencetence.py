import yaml
import pathlib
import re
import tqdm


def split_file_content_to_sentence(file, save_path, sentence_num, step_size):
    file = pathlib.Path(file)
    with open(file, 'r') as r:
        # CME需要的是第0行
        # content = r.readlines()[0]
        content = r.readlines()[2]
    sentences = re.split('(。|！|？|；)', content)

    print(sentences)

    results = 0
    s = ''
    for i in range(0, len(sentences), step_size * 2):
        save_file = save_path.joinpath(file.stem + "_" + str(i // 2) + '.txt')
        s = s + ''.join(sentences[i:i + sentence_num * 2])
        #and len(s)>30
        if len(s) != 0 and s != '\n' and s != '。' and s != '！' and s != '？' and s != '；' and s != ',' and s != '?':
            with open(save_file, 'w') as w:
                w.write(s)
            results += 1
            s = ''


    return results




def split_files_content_to_sentence(files, save_path, sentence_num, step_size):
    results = 0
    save_path = pathlib.Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    for file in tqdm.tqdm(files):
        file = pathlib.Path(file)
        result = split_file_content_to_sentence(file, save_path, sentence_num, step_size)
        #print(file)
        #print(save_path)
        if result != len(list(save_path.glob(file.stem + "_*"))):
            #print(file)
            input("enter")
        results += result
    #print(len(list(save_path.glob('*'))))
    #print(results)
    return results


def get_files(root_dir):
    return list(pathlib.Path(root_dir).glob('*'))


if __name__ == "__main__":
    config_file = '../configs_2_simple/sentence-4.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # 将文章切分为句子
    origin_doc_path = config['origin_doc_path']
    sentence_doc_path = config['init_sentence_doc_path']
    sentence_num = int(config['init_sentence_num'])
    step_size = int(config['init_step_size'])

    files = get_files(origin_doc_path)

    split_files_content_to_sentence(files, sentence_doc_path, sentence_num, step_size)
