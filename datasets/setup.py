import nltk
import os
import ssl

# 创建nltk数据目录
nltk_data_path = os.path.join(os.getcwd(), 'nltk')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 设置下载路径
nltk.data.path.append(nltk_data_path)

# 下载必要的NLTK资源
resources = [
    'wordnet',
    'stopwords',
    'punkt',
    'averaged_perceptron_tagger'
]

for resource in resources:
    print(f"正在下载 {resource}...")
    nltk.download(resource, download_dir=nltk_data_path)

print("NLTK资源下载完成！现在可以运行main.py了。") 