import pandas as pd

def Intent_Dict():
    intent_df = pd.read_table('intent_label.txt').rename({'UNK':'intent'}, axis=1)
    labelName = intent_df.intent.unique()# 全部label列表
    label_index_dict = dict(zip(labelName, range(len(labelName))))  # 实现文本label 与index的映射 hint : zip dict
    dict_new = {value: key for key, value in label_index_dict.items()}
    return dict_new

def replace(list, dictionary):
    return [dictionary.get(item, item) for item in list]

def Num2Intent(label_index_dict, predict):
    num_list=predict['ERNIE(预训练模型)'].to_list()
    num_list = replace(num_list, label_index_dict)
    predict['ERNIE(预训练模型)'] = num_list

def compare(compare1, compare2, outname, predict,compare3=None):
    output = predict
    drop_list = []
    if compare3 == None:
        for i in range(len(output)):
            if output[compare1][i] == output[compare2][i]:
                drop_list.append(i)
    else :
        for i in range(len(output)):
            if not (output[compare1][i] == output[compare2][i] and output[compare1][i] != output[compare3][i]):
                drop_list.append(i)
    output = output.drop(drop_list)
    output.to_csv(outname+'.csv',index=False)

if __name__ == '__main__':
    # label_index_dict = Intent_Dict()
    predict = pd.read_csv('predict.csv')
    # Num2Intent(label_index_dict, predict)
    # predict.to_csv('predict.csv')
    compare('intent', 'ERNIE(预训练模型)', 'ERNIE(预训练模型)_错误', predict)
    compare('intent', 'Bi-LSTM(非预训练模型)', 'Bi-LSTM(非预训练模型)_错误', predict)
    compare('intent','ERNIE(预训练模型)', 'ERNIE(预训练模型)_正确_Bi-LSTM(非预训练模型)_错误', predict, 'Bi-LSTM(非预训练模型)')
    compare('intent', 'Bi-LSTM(非预训练模型)', 'ERNIE(预训练模型)_错误_Bi-LSTM(非预训练模型)_正确', predict, 'ERNIE(预训练模型)')
