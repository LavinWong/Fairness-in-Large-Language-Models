import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import string
import re
import os
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import inspect


# Set the path to your local nltk folder/
nltk_data_path = os.path.join(os.getcwd(), 'nltk')
nltk.data.path.append(nltk_data_path)

nlp = spacy.load("en_core_web_sm")

unwanted = {
    "he", "his", "her", "she", "him", "man", "woman", "men", "women", "spokesman", 
    "wife", "himself", "son", "mother", "father", "chairman", "daughter", "husband", 
    "guy", "girls", "girl", "boy", "boys", "brother", "spokeswoman", "female", 
    "sister", "male", "herself", "brothers", "dad", "actress", "mom", "sons", 
    "girlfriend", "daughters", "lady", "boyfriend", "sisters", "mothers", "king", 
    "businessman", "grandmother", "grandfather", "deer", "ladies", "uncle", "males", 
    "congressman", "grandson", "bull", "queen", "businessmen", "wives", "widow", 
    "nephew", "bride", "females", "aunt", "prostate cancer", "lesbian", "chairwoman", 
    "fathers", "moms", "maiden", "granddaughter", "younger brother", "lads", "lion", 
    "gentleman", "fraternity", "bachelor", "niece", "bulls", "husbands", "prince", 
    "colt", "salesman", "hers", "dude", "beard", "filly", "princess", "lesbians", 
    "councilman", "actresses", "gentlemen", "stepfather", "monks", "ex-girlfriend", 
    "lad", "sperm", "testosterone", "nephews", "maid", "daddy", "mare", "fiancé", 
    "fiancée", "kings", "dads", "waitress", "maternal", "heroine", "nieces", 
    "girlfriends", "sir", "stud", "mistress", "lions", "estranged wife", "womb", 
    "grandma", "maternity", "estrogen", "ex-boyfriend", "widows", "gelding", "diva", 
    "teenage girls", "nuns", "czar", "ovarian cancer", "countrymen", "teenage girl", 
    "penis", "bloke", "nun", "brides", "housewife", "spokesmen", "suitors", 
    "menopause", "monastery", "motherhood", "brethren", "stepmother", "prostate", 
    "hostess", "twin brother", "schoolboy", "brotherhood", "fillies", "stepson", 
    "congresswoman", "uncles", "witch", "monk", "viagra", "paternity", "suitor", 
    "sorority", "macho", "businesswoman", "eldest son", "gal", "statesman", 
    "schoolgirl", "fathered", "goddess", "hubby", "stepdaughter", "blokes", 
    "dudes", "strongman", "uterus", "grandsons", "studs", "mama", "godfather", 
    "hens", "hen", "mommy", "estranged husband", "elder brother", "boyhood", 
    "baritone", "grandmothers", "grandpa", "boyfriends", "feminism", "countryman", 
    "stallion", "heiress", "queens", "witches", "aunts", "semen", "fella", 
    "granddaughters", "chap", "widower", "salesmen", "convent", "vagina", 
    "beau", "beards", "handyman", "twin sister", "maids", "gals", "housewives", 
    "horsemen", "obstetrics", "fatherhood", "councilwoman", "princes", "matriarch", 
    "colts", "ma", "fraternities", "pa", "fellas", "councilmen", "dowry", 
    "barbershop", "fraternal", "ballerina"}

def get_wordnet_pos(spacy_tag):
    if spacy_tag in ['ADJ']:
        return 'a'  # Adjective
    elif spacy_tag in ['VERB']:
        return 'v'  # Verb
    elif spacy_tag in ['NOUN', 'PROPN']:
        return 'n'  # Noun
    elif spacy_tag in ['ADV']:
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun

def remove_punctuation(sentence):
    """Remove punctuation from a sentence."""
    translator = str.maketrans('', '', string.punctuation + "[](){}")
    return sentence.translate(translator)

def remove_numbers(sentence):
    """Remove numbers from a sentence."""
    return re.sub(r'\d+', '', sentence)

def preprocess_sentence_for_keywords(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    doc = nlp(sentence)
    filtered_words_pos = [(token.text, token.pos_) for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON', 'ADJ']]
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in filtered_words_pos]
    lemmatized_words = [word.lower() for word in lemmatized_words]
    lemmatized_words = [remove_numbers(remove_punctuation(word)) for word in lemmatized_words]
    final_words = [word for word in lemmatized_words if word and word not in stop_words and word.isalnum()]

    return final_words

def cosine_distance(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 1.0
    cos_sim = 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

def generate_keywords_from_sentences(sentences, k):
    all_candidates = set()

    sentences = [sentence for sentence in sentences if isinstance(sentence, str) and sentence.strip()]

    for sentence in sentences:
        preprocessed_words = preprocess_sentence_for_keywords(sentence)
        
        all_candidates.update(preprocessed_words)
    
    all_candidates = all_candidates.difference(unwanted)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
    tfidf_scores_dict = dict(zip(tfidf_feature_names, tfidf_scores))
    
    candidate_tfidf_scores = {candidate: tfidf_scores_dict.get(candidate, 0) for candidate in all_candidates}
    top_keywords = sorted(candidate_tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    keywords = [word for word, score in top_keywords]

    return keywords

def visualize_gender_bias_tfidf(sentences, keywords):
    """
    可视化文本中的性别偏见，基于TF-IDF向量空间表示。
    
    参数:
        sentences: 文本句子列表
        keywords: 需要分析性别偏见的关键词列表
    """
    # 获取当前数据集名称用于图表标题和保存
    frame = inspect.currentframe()
    try:
        parent_frame = frame.f_back
        dataset_name = parent_frame.f_locals.get('dataset_name', 'unknown_dataset')
    except:
        dataset_name = 'unknown_dataset'
    finally:
        del frame  # 避免循环引用
    
    # 创建图表
    plt.figure(figsize=(12, 12))

    # 步骤1: 使用TF-IDF将文本转换为向量表示
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    # 过滤掉已知的性别相关词汇，留下中性词汇用于偏见计算
    words_for_bias_calculation = [word for word in tfidf_feature_names if word not in unwanted]

    # 步骤2: 定义性别词对，用于确定向量空间中的性别方向
    gender_pairs = [("she", "he"), ("her", "his"), ("woman", "man"), ("mary", "john"), ("herself", "himself"), ("daughter", "son"), ("mother", "father"), ("gal", "guy"), ("girl", "boy"), ("female", "male")]

    # 计算每对性别词的向量差异
    pair_diffs = []
    male_vector_list, female_vector_list = [], []
    for (female_word, male_word) in gender_pairs:
        try:
            # 获取性别词的向量表示
            female_idx = tfidf_vectorizer.vocabulary_[female_word]
            male_idx = tfidf_vectorizer.vocabulary_[male_word]
            
            female_vector = tfidf_matrix[:, female_idx].toarray().flatten()
            male_vector = tfidf_matrix[:, male_idx].toarray().flatten()
            
            # 计算女性词-男性词的向量差，表示从男性到女性的方向
            diff_vector = female_vector - male_vector
            pair_diffs.append(diff_vector)

            # 保存各个性别词的向量用于后续计算平均向量
            male_vector_list.append(male_vector)
            female_vector_list.append(female_vector)
        except KeyError:
            print(f"'{female_word}' or '{male_word}' not found in the vocabulary.")

    # 步骤3: 使用PCA提取主要的性别方向向量
    if len(pair_diffs) > 0:
        pca = PCA(n_components=1)
        pca.fit(pair_diffs)
        
        gender_direction = pca.components_[0]  # 得到主要的性别方向
    else:
        print("No definitional pairs found in the vocabulary. Cannot calculate gender direction.")
        return

    # 步骤4: 计算男性和女性词汇的平均向量作为参考点
    male_average_vector = np.mean(male_vector_list, axis=0)
    female_average_vector = np.mean(female_vector_list, axis=0)

    # 步骤5: 计算所有中性词的性别偏见
    distances_from_male, distances_from_female = [], []
    direct_bias = []
    all_words_direct_bias = {}
    for word in words_for_bias_calculation:
        word_index = tfidf_vectorizer.vocabulary_[word]
        word_vector = tfidf_matrix[:, word_index].toarray().flatten()

        # 计算与男性/女性平均向量的余弦距离
        dist_from_male = 1 - cosine_similarity([word_vector], [male_average_vector])[0][0]
        dist_from_female = 1 - cosine_similarity([word_vector], [female_average_vector])[0][0]

        distances_from_male.append(dist_from_male)
        distances_from_female.append(dist_from_female)

        # 计算词向量在性别方向上的投影大小，作为直接偏见度量
        cosine_sim = np.abs(np.dot(word_vector, gender_direction) / 
                            (np.linalg.norm(word_vector) * np.linalg.norm(gender_direction)))
        direct_bias.append(cosine_sim)
        all_words_direct_bias[word] = cosine_sim 
    
    # 计算整体直接偏见指标
    overall_direct_bias = np.mean(direct_bias)
    print("Overall Direct Bias (excluding unwanted words):", overall_direct_bias)

    # 步骤6: 专门为关键词计算性别偏见(用于可视化)
    distances_from_male, distances_from_female = [], []
    for keyword in keywords:
        if keyword in tfidf_vectorizer.vocabulary_:
            keyword_index = tfidf_vectorizer.vocabulary_[keyword]
            keyword_vector = tfidf_matrix[:, keyword_index].toarray().flatten()

            # 计算关键词与男性/女性平均向量的余弦距离
            dist_from_male = 1 - cosine_similarity([keyword_vector], [male_average_vector])[0][0]
            dist_from_female = 1 - cosine_similarity([keyword_vector], [female_average_vector])[0][0]

            distances_from_male.append(dist_from_male)
            distances_from_female.append(dist_from_female)
    
    # 步骤7: 设置可视化的范围
    min_value = min(distances_from_male + distances_from_female)
    max_value = max(distances_from_male + distances_from_female)
    plt.xlim([min_value-0.005, max_value+0.005])
    plt.ylim([min_value-0.005, max_value+0.005])

    # 绘制对角线，表示男女中性位置
    x_vals = np.linspace(-1.15, 1.15, 400)
    y_vals = x_vals
    plt.plot(x_vals, y_vals, color='black', linewidth=1)

    # 标记偏男性和偏女性的区域
    plt.fill_between(x_vals, y_vals, 1.15, color='red', alpha=0.1)  # 偏男性区域
    plt.fill_between(x_vals, -1.15, y_vals, color='blue', alpha=0.1)  # 偏女性区域

    # 步骤8: 绘制关键词散点图并标记性别偏向
    masculism_words = []
    feminism_words = []

    for i, word in enumerate(keywords):
        if distances_from_male[i] == distances_from_female[i]:
            color = 'black'  # 中性词
        elif distances_from_male[i] < distances_from_female[i]:
            distances_from_male[i] 
            masculism_words.append(word)
            color = 'red'  # 偏男性
        else: 
            distances_from_female[i]
            feminism_words.append(word)
            color = 'blue'  # 偏女性
        plt.scatter(distances_from_male[i], distances_from_female[i], c=color)
        plt.text(distances_from_male[i]+0.001, distances_from_female[i], word, fontsize=11)

    # 输出偏向词汇列表
    print("Masculism words: ", masculism_words)
    print("Feminism words: ", feminism_words)

    # 步骤9: 添加图表标题和轴标签
    plt.xlabel('Cosine Distance to Average Male Terms\' Embeddings')
    plt.ylabel('Cosine Distance to Average Female Terms\' Embeddings')
    plt.title(f'Gender Bias Visualization for Selected Keywords - {dataset_name}')
    plt.grid(True)
    
    # 步骤10: 保存可视化结果
    if not os.path.exists('./bias_visuliazation'):
        os.makedirs('./bias_visuliazation')
    
    plt.savefig(f'./bias_visuliazation/{dataset_name}.png')
    print(f"图片已保存至: ./bias_visuliazation/{dataset_name}.png")
    
    # 显示图片并阻止自动关闭
    plt.show(block=True)
