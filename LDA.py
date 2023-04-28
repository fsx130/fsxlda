import os
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def content_deal(content):  # 语料预处理，进行断句，去除一些广告和无意义内容
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        content = content.replace(a, '')
    return content

def read_files(path):
    texts = []
    labels = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='GB18030') as f:
            content = f.read()
            content = content_deal(content)
            total_len = len(content)
            size = int(total_len // 13)
            for i in range(13):
                start = i * size
                end = (i + 1) * size
                selected_text = content[start:end][:5000]
                if selected_text:
                    texts.append(content)
                    labels.append(filename)
    return texts, labels

def tokenize(texts, unit):#分词
    tokenized_texts = []
    for text in texts:
        if unit == 'word':
            tokens = list(jieba.cut(text))
        elif unit == 'character':
            tokens = list(text)
        tokenized_texts.append(tokens)
    return tokenized_texts

def prepare_data(path, n_topics, unit):
    texts, labels = read_files(path)
    tokenized_texts = tokenize(texts, unit)
    #文档的词袋向量表示
    count_vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
    X = count_vectorizer.fit_transform(tokenized_texts)
    #训练模型
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='batch', max_iter=20, random_state=0)
    lda.fit(X)
    # Transform texts to topic distributions
    X_topics = lda.transform(X)
    return X_topics, labels

def evaluate_classification(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)
    # Train and test a logistic regression classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

# Test different numbers of topics and units
folder_path = "C:\\Users\\feng\\Desktop\\NPLwork\\corpus"
n_topics_list = [10, 100, 200]
unit_list = ['word', 'character']

for n_topics in n_topics_list:
    for unit in unit_list:
        X, y = prepare_data(folder_path, n_topics, unit)
        acc = evaluate_classification(X, y)
        print(f"n_topics={n_topics}, unit={unit}, accuracy={acc:.3f}")
