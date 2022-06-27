Intend Detection
===
意图识别：传统方法和深度学习方法

快速开始
---
### 数据集
* ATIS：英文数据集，训练数据4978条，测试数据888条，类别22个
* SNIPS：英文数据集，训练数据13784条，测试数据700条，类别7个

### SVM
* python -u train.py -dd {$DATE}

### LR
* python -u train.py -dd {$DATE}

### Stack-Propagation
* python -u train.py -dd {$DATE}

### Bi-model with decoder
* python -u train.py 

### Bi-LSTM
* python -u train.py

### JointBERT
* python -u main.py --task {$DATE} --model_dir {$MODEL DIR}

### ERNIE
* python -u train.py --task {$DATE} --model_dir {$MODEL DIR}

测试
---
* 测试输出训练时间、测试时间、训练准确率、测试准确率
* 训练时间基于从开始训练到结束训练的时间
* 测试时间基于测试一次训练集的时间,传统模型是在CPU的条件下进行测试，深度学习模型是在GPU的条件下batch_size为1的条件下进行测试
* 训练准确率基于训练集准确率
* 测试准确率基于测试集准确率

模型介绍
---
* SVM：支持向量机模型，基于TF-IDF或wordvec
* LR：逻辑回归模型，基于TF-IDF或wordvec
* Stack-Propagation：https://aclanthology.org/D19-1214.pdf
* Bi-model with decoder：https://aclanthology.org/N18-2050.pdf
* Bi-LSTM：单层双向LSTM模型，基于最后一个时间步结果
* JointBERT：https://arxiv.org/pdf/1902.10909v1.pdf
* ERNIE：ERNIE预训练模型，基于[CLS]位判断

