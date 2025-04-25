import tools as tools
from data_loader import DataLoader
import json

def print_invalid_command():
    print("Invalid command!")

# 在这里设置参数，而不是从命令行获取
# 设置模式: "visualize" 或 "keywords"
mode = "visualize"
# 设置数据集: "winobias", "winogender", "gap"等
dataset_name = "winogender"
# 如果mode是"keywords"，可以设置要提取的关键词数量
top_k = 100

command = 1 if mode == "keywords" else 0
dataset = []
keywords_dict = {}

with open('./keywords.json') as f:
    keywords_dict = json.load(f)

if dataset_name == "winobias":
    dataset = DataLoader.load_dataset_winobias()
elif dataset_name == "winobias+":
    dataset = DataLoader.load_dataset_winobiasplus()
elif dataset_name == "winogender":
    dataset = DataLoader.load_dataset_winogender()
elif dataset_name == "gap":
    dataset = DataLoader.load_dataset_gap()
elif dataset_name == "bug":
    dataset = DataLoader.load_dataset_bug()
elif dataset_name == "stereoset":
    dataset = DataLoader.load_dataset_stereset()
elif dataset_name == "becpro":
    dataset = DataLoader.load_dataset_becpro()
elif dataset_name == "honest":
    dataset = DataLoader.load_dataset_honest()
elif dataset_name == "crowspairs":
    dataset = DataLoader.load_dataset_crowspairs()
elif dataset_name == "eec":
    dataset = DataLoader.load_dataset_eec()
elif dataset_name == "panda":
    dataset = DataLoader.load_dataset_panda()
elif dataset_name == "redditbias":
    dataset = DataLoader.load_dataset_redditbias()
elif dataset_name == "winoqueer":
    dataset = DataLoader.load_dataset_winoqueer()
elif dataset_name == "bold":
    dataset = DataLoader.load_dataset_bold()
elif dataset_name == "holisticbias":
    dataset = DataLoader.load_dataset_holisticbias()
elif dataset_name == "bbq":
    dataset = DataLoader.load_dataset_bbq()
elif dataset_name == "unqover":
    dataset = DataLoader.load_dataset_unqover()
elif dataset_name == "ceb":
    dataset = DataLoader.load_dataset_ceb()
else:
    print_invalid_command()
    exit()

keywords = keywords_dict[dataset_name]
if command:  # keywords 模式
    top_k_keywords = tools.generate_keywords_from_sentences(dataset, top_k)
    print(f'Top {top_k} keywords of {dataset_name}: ', top_k_keywords)
else:  # visualize 模式
    if len(keywords) == 0:
        keywords = tools.generate_keywords_from_sentences(dataset, 100)  
    # tools.visualize_gender_bias_tfidf(dataset, keywords)
    tools.visualize_gender_bias_word2vec(dataset, keywords)
