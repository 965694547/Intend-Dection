from torch import optim
import numpy as np
import torch
import time
import pandas as pd

import utils
from utils import get_chunks
from config import device
import config as cfg
from data2index_ver2 import train_data, test_data
from model import *

train_start=time.time()
epoch_num = cfg.total_epoch

#slot_model = Slot().to(device)
intent_model = Intent().to(device)
model_dict=torch.load('./model_intent_best.ckpt')
intent_model.load_state_dict(model_dict)

print(intent_model)

correct_num=0
total_test = len(train_data)
predict = []
for batch_index, data in enumerate(utils.get_batch(train_data, batch_size=64)):
    sentence_test, real_len_test, intent_label_test = data
    # print(sentence[0].shape, real_len.shape, slot_label.shape)
    x_test = torch.tensor(sentence_test).to(device)

    # mask_test = utils.make_mask(real_len_test, batch=1).to(device)
    # Slot model generate hs_test and intent model generate hi_test
    # hs_test = slot_model.enc(x_test)
    hi_test = intent_model.enc(x_test)

    intent_logits_test = intent_model.dec(hi_test)
    log_intent_logits_test = F.log_softmax(intent_logits_test, dim=-1)
    res_test = torch.argmax(log_intent_logits_test, dim=-1)

    res_test_list = res_test.cpu().numpy()
    predict.extend(res_test_list)

    com = np.array(res_test_list - intent_label_test)
    cnt_array = np.where(com, 0, 1)
    correct_num += np.sum(cnt_array)
    #if res_test.item() == intent_label_test[0]:
    #    correct_num += 1
print('Test Acc: {:.4f}'.format(100.0 * correct_num / total_test))
output = pd.DataFrame(data=predict)
output.to_csv('error.csv', index=None, header=None)



