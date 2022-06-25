import pandas as pd
'''
import nltk #英文分词
import jieba #中文分词

# 分词
def query_cut_Chinese(query):
    return list(jieba.cut(query))
def query_cut_English(query):
    return list(nltk.word_tokenize(query))
'''

# 写入txt文件
def write_file(mode):
    f_in = open(mode + '.txt','r')
    f_out = open(mode, 'w')
    text = ''
    intents=[]
    for line in f_in.readlines():
        items = line.strip().split()

        if len(items) == 1:
            text=text+'<=> '+items[0]
            intents.append(items)
            f_out.write(text+'\n')

            text = ''
        elif len(items) == 2:
            text=text+items[0].strip()+':'+'0'+' '
    f_in.close()
    f_out.close()
    return intents

if __name__ == '__main__':
    #train_df = pd.read_csv('StackPropagation-SLU-master/data/dh_nh/train.csv', header=0, index_col=None)
    #test_df = pd.read_csv('StackPropagation-SLU-master/data/dh_nh/test.csv', header=0, index_col=None)

    #labelName = train_df.intent.unique()# 全部label列表
    #label_index_dict = dict(zip(labelName, range(len(labelName))))# 实现文本label 与index的映射 hint : zip dict

    # 将dataframe 中文本label转换为数字。 hint:  map
    #train_df["labelIndex"] = train_df.intent.map(lambda x: label_index_dict.get(x))
    #test_df["labelIndex"] = test_df.intent.map(lambda x: label_index_dict.get(x))

    #for i in range(len(test_df)):
    #    try:
    #        label_index_dict[test_df['intent'][i]]
    #        continue
    #    except:
    #        test_df=test_df.drop(i)

    # 分词
    #train_df["queryCut"] = train_df["text"].apply(query_cut_Chinese)
    #test_df["queryCut"] = test_df["text"].apply(query_cut_Chinese)
    #cut_idx = int(round(0.05 * train_df.shape[0]))
    #dev_df, train_df = train_df.iloc[:cut_idx], train_df.iloc[cut_idx:]
    #write_file(dev_df, 'dev')
    intend=write_file('train')
    write_file('test')
    data = pd.DataFrame(intend)
    labelName = data[0].unique()
    f = open('vocab', 'w')
    for item in labelName:
        f.write(item+ '\n')
    f.close()



