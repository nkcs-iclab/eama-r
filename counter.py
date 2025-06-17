import os


def count_subdirectories(folder_path):
    count = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            count += 1
    return count


# 示例使用
folder_path = '/home/fengmingjian/src/emb/emb-patent-word-scentence-ENG-file'
print(count_subdirectories(folder_path))