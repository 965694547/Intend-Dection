# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。

import numpy as np
import pandas as pd
import json #中文分词
import nltk #英文分词
import jieba
import collections
import time

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

test_time=1# 测试次数
# 对数据中的文本进行分词 hint:  jieba.cut
# jieba.cut 返回一个generator, 需要进行转换 hint: list
def query_cut_Chinese(query):
    return list(jieba.cut(query))
def query_cut_English(query):
    return list(nltk.word_tokenize(query))

# 使用停词进一步过滤分词
def rm_stop_word(wordList):
    new_wordList = [word for word in wordList if word not in stopWords]
    return new_wordList
# 过了低频词
def rm_low_fre_word(query):
    new_query = [word for word in query if word in highFreWords]
    return new_query


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 读取数据至DataFrame中
    '''
    load_f=open("数据集/ATIS/ATIS_dataset-master/data/standard_format/rasa/train.json", 'r')
    load_dict = json.load(load_f)['rasa_nlu_data']['common_examples']
    train_df=pd.DataFrame.from_dict(load_dict)
    train_df.pop('entities')

    load_f=open("数据集/ATIS/ATIS_dataset-master/data/standard_format/rasa/test.json", 'r')
    load_dict = json.load(load_f)['rasa_nlu_data']['common_examples']
    test_df=pd.DataFrame.from_dict(load_dict)
    test_df.pop('entities')
    '''
    train_df=pd.read_table('数据集/SNIPS/snips/train/seq.txt',header=None).rename({0:'text'}, axis=1)
    train_df['intent']=pd.read_table('数据集/SNIPS/snips/train/label.txt',header=None)

    test_df=pd.read_table('数据集/SNIPS/snips/test/seq.txt',header=None).rename({0:'text'}, axis=1)
    test_df['intent']=pd.read_table('数据集/SNIPS/snips/test/label.txt',header=None)

    labelName = train_df.intent.unique()# 全部label列表
    label_index_dict = dict(zip(labelName, range(len(labelName))))# 实现文本label 与index的映射 hint : zip dict

    # 将dataframe 中文本label转换为数字。 hint:  map
    train_df["labelIndex"] = train_df.intent.map(lambda x: label_index_dict.get(x))
    test_df["labelIndex"] = test_df.intent.map(lambda x: label_index_dict.get(x))

    for i in range(len(test_df)):
        try:
            label_index_dict[test_df['intent'][i]]
            continue
        except:
            test_df=test_df.drop(i)

    '''
    cut_idx = 888
    dev_df = train_df.iloc[:cut_idx]
    train_df=train_df[cut_idx:]
    train_df.to_csv('train.csv')
    train_df.to_csv('train.csv')
    test_df.to_csv('test.csv')
    dev_df.to_csv('dev.csv')
    '''

    # 分词
    #train_df["queryCut"] = train_df["text"].apply(query_cut_Chinese)
    #test_df["queryCut"] = test_df["text"].apply(query_cut_Chinese)
    train_df["queryCut"] = train_df["text"].apply(query_cut_English)
    test_df["queryCut"] = test_df["text"].apply(query_cut_English)

    # 读取停用词
    stopWords=pd.read_json("stopwords-json-master/dist/en.json")
    #stopWords=pd.read_json("/Users/guozhifang/Desktop/Intend Detection Program/SVM/stopwords-json-master/dist/zh.json")
    train_df["queryCutRMStopWord"] = train_df["queryCut"].apply(rm_stop_word)
    test_df["queryCutRMStopWord"] = test_df["queryCut"].apply(rm_stop_word)

    # 计算词频
    allWords = [word for query in train_df.queryCutRMStopWord for word in query]  # 所有词组成的列表
    freWord = dict(collections.Counter(allWords))  # 统计词频，一个字典，键为词，值为词出现的次数

    # 过滤低频词
    highFreWords = [word for word in freWord.keys() if freWord[word] > 3]  # 词频超过3的词列表
    def rm_low_fre_word(query):
        # TODO
        new_query = [word for word in query if word in highFreWords]
        return new_query

    # 去除低频词
    train_df["queryFinal"] = train_df["queryCutRMStopWord"].apply(rm_low_fre_word)
    test_df["queryFinal"] = test_df["queryCutRMStopWord"].apply(rm_low_fre_word)

    '''
    train_df.to_csv('snips_train_df.csv')
    test_df.to_csv('snips_test_df.csv')
    
    train_df=pd.read_csv('snips_train_df.csv',index_col=0)
    test_df=pd.read_csv('snips_test_df.csv',index_col=0)
    '''

    # 将分词且过滤后的文本数据转化为tfidf 形式：
    trainText = [' '.join(query) for query in train_df["queryFinal"]]
    testText = [' '.join(query) for query in test_df["queryFinal"]]
    allText = trainText + testText

    # sklearn tfidf vector fit_transform
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(allText))

    # 切分数据集 hint sklearn train_test_split()
    trainLen = len(train_df)
    train_x_tfidf = tfidf.toarray()[0:trainLen]
    test_x_tfidf = tfidf.toarray()[trainLen:]
    train_y_tfidf = train_df["labelIndex"]
    test_y_tfidf = test_df["labelIndex"]

    # 训练词向量
    from gensim.models import word2vec
    file = open('word2vec.txt', 'w');
    file.write(str(trainText))
    file.close()
    sentences = word2vec.Text8Corpus('word2vec.txt')
    model = word2vec.Word2Vec(sentences).wv
    vocabulary = model.key_to_index
    def sentence2vec(query):
        result_array = np.zeros(len(model['i']))
        if len(query):
            for word in query:
                if word not in vocabulary:
                    rand_array = -1 + 2 * np.random.random(
                        size=len(model['i']))  # randomly generate an array between -1 and 1
                    result_array = np.vstack((result_array, rand_array))
                else:
                    result_array = np.vstack((result_array, model.get_vector(word)))
            return result_array.mean(axis=0)  # get the average value
        else:
            return np.zeros(len(model['i']))

    # 将转换为词向量的数据， 切分为训练集， 验证集
    train_x_vec = np.vstack(train_df["queryCutRMStopWord"].apply(sentence2vec))
    test_x_vec = np.vstack(test_df["queryCutRMStopWord"].apply(sentence2vec))
    train_y_vec = train_df["labelIndex"]
    test_y_vec = test_df["labelIndex"]

    print('线性SVM(word2vec)')
    start = time.perf_counter()
    # 使用embeding 特征建立线性SVM模型
    word2vecLinearSVM = SVC(kernel='linear')
    word2vecLinearSVM.fit(train_x_vec, train_y_vec)
    end = time.perf_counter()
    # 输出模型结果， accuracy,  F1_score
    print('train accuracy %s' % metrics.accuracy_score(train_y_vec, word2vecLinearSVM.predict(train_x_vec)))
    print('test accuracy %s' % metrics.accuracy_score(test_y_vec, word2vecLinearSVM.predict(test_x_vec)))
    print("train time %s s" % (str(end - start)))
    start = time.perf_counter()
    word2vecLinearSVM.predict(train_x_vec)
    end = time.perf_counter()
    print("test time %s s" % (str(end - start)))
    '''
    # 效率检测
    total_time=0
    for i in range(test_time):
        start=time.perf_counter()
        word2vecLinearSVM = SVC(kernel='linear')
        word2vecLinearSVM.fit(train_x_vec, train_y_vec)
        end= time.perf_counter()
        total_time=total_time+end-start
    print("train time %s s" % (str(total_time/test_time)))
    word2vecLinearSVM = SVC(kernel='linear')
    word2vecLinearSVM.fit(train_x_vec, train_y_vec)
    total_time = 0
    for i in range(test_time):
        start=time.perf_counter()
        word2vecLinearSVM.predict(train_x_vec)
        end= time.perf_counter()
        total_time=total_time+end-start
    print("test time %s s" % (str(total_time/test_time)))
    '''

    print('rfb SVM(word2vec)')
    start = time.perf_counter()
    # 使用embedding  特征建立`rbf` SVM模型
    word2vecKernelizedSVM = SVC(kernel='rbf')
    word2vecKernelizedSVM.fit(train_x_vec, train_y_vec)
    end = time.perf_counter()
    # 输出模型结果， accuracy,  F1_score
    print('train accuracy %s' % metrics.accuracy_score(train_y_vec, word2vecKernelizedSVM.predict(train_x_vec)))
    print('test accuracy %s' % metrics.accuracy_score(test_y_vec, word2vecKernelizedSVM.predict(test_x_vec)))
    print("train time %s s" % (str(end - start)))
    start = time.perf_counter()
    word2vecLinearSVM.predict(train_x_vec)
    end = time.perf_counter()
    print("test time %s s" % (str(end - start)))
    '''
    # 效率检测
    total_time=0
    for i in range(test_time):
        start=time.perf_counter()
        word2vecKernelizedSVM = SVC(kernel='rbf')
        word2vecKernelizedSVM.fit(train_x_vec, train_y_vec)
        end= time.perf_counter()
        total_time=total_time+end-start
    print("train time %s s" % (str(total_time/test_time)))
    word2vecKernelizedSVM = SVC(kernel='rbf')
    word2vecKernelizedSVM.fit(train_x_vec, train_y_vec)
    total_time = 0
    for i in range(test_time):
        start=time.perf_counter()
        word2vecLinearSVM.predict(train_x_vec)
        end= time.perf_counter()
        total_time=total_time+end-start
    print("test time %s s" % (str(total_time/test_time)))
    '''

    print('线性SVM(TF-IDF)')
    # 使用tfidf 特征建立线性SVM模型 hint: SVC()
    start = time.perf_counter()
    tfidfLinearSVM = SVC(kernel='linear')
    tfidfLinearSVM.fit(train_x_tfidf, train_y_tfidf)
    end = time.perf_counter()
    # 输出模型结果， accuracy,  F1_score
    print('train accuracy %s' % metrics.accuracy_score(train_y_tfidf, tfidfLinearSVM.predict(train_x_tfidf)))
    print('test accuracy %s' % metrics.accuracy_score(test_y_tfidf, tfidfLinearSVM.predict(test_x_tfidf)))
    print("train time %s s" % (str(end - start)))
    start = time.perf_counter()
    tfidfLinearSVM.predict(train_x_tfidf)
    end = time.perf_counter()
    print("test time %s s" % (str(end - start)))

    '''
    # 效率检测
    total_time=0
    for i in range(test_time):
        start=time.perf_counter()
        tfidfLinearSVM = SVC(kernel='linear')
        tfidfLinearSVM.fit(train_x_tfidf, train_y_tfidf)
        end= time.perf_counter()
        total_time=total_time+end-start
    print("train time %s s" % (str(total_time/test_time)))
    tfidfLinearSVM = SVC(kernel='linear')
    tfidfLinearSVM.fit(train_x_tfidf, train_y_tfidf)
    total_time = 0
    for i in range(test_time):
        start=time.perf_counter()
        tfidfLinearSVM.predict(train_x_tfidf)
        end= time.perf_counter()
        total_time=total_time+end-start
    print("test time %s s" % (str(total_time/test_time)))
    '''

    print('rbf SVM(TF-IDF)')
    start = time.perf_counter()
    # 使用tfidf 特征建立`rbf` SVM 模型
    tfidfKernelizedSVM = SVC(kernel='rbf')
    tfidfKernelizedSVM.fit(train_x_tfidf, train_y_tfidf)
    end = time.perf_counter()
    # 输出模型结果， accuracy,  F1_score
    print('train accuracy %s' % metrics.accuracy_score(train_y_tfidf, tfidfKernelizedSVM.predict(train_x_tfidf)))
    print('test accuracy %s' % metrics.accuracy_score(test_y_tfidf, tfidfKernelizedSVM.predict(test_x_tfidf)))
    print("train time %s s" % (str(end - start)))
    start = time.perf_counter()
    tfidfLinearSVM.predict(train_x_tfidf)
    end = time.perf_counter()
    print("test time %s s" % (str(end - start)))
    '''
    # 效率检测
    total_time=0
    for i in range(test_time):
        start=time.perf_counter()
        tfidfKernelizedSVM = SVC(kernel='rbf')
        tfidfKernelizedSVM.fit(train_x_tfidf, train_y_tfidf)
        end= time.perf_counter()
        total_time=total_time+end-start
    print("train time %s s" % (str(total_time/test_time)))
    tfidfKernelizedSVM = SVC(kernel='rbf')
    tfidfKernelizedSVM.fit(train_x_tfidf, train_y_tfidf)
    total_time = 0
    for i in range(test_time):
        start=time.perf_counter()
        tfidfLinearSVM.predict(train_x_tfidf)
        end= time.perf_counter()
        total_time=total_time+end-start
    print("test time %s s" % (str(total_time/test_time)))
    '''





