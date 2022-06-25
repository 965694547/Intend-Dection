from torch import optim
import numpy as np
import torch
import time

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

#print(slot_model)
print(intent_model)

#slot_optimizer = optim.Adam(slot_model.parameters(), lr=cfg.learning_rate)       # optim.Adamax
intent_optimizer = optim.Adam(intent_model.parameters(), lr=cfg.learning_rate)   # optim.Adamax

best_correct_num = 0
best_epoch = -1
best_F1_score = 0.0
#best_epoch_slot = -1
for epoch in range(epoch_num):
    #slot_loss_history = []
    intent_loss_history = []
    correct_num = 0
    total_test = len(train_data)
    for batch_index, data in enumerate(utils.get_batch(train_data)):

	    # Preparing data
        sentence, real_len, intent_label = data

        #mask = utils.make_mask(real_len).to(device)
        x = torch.tensor(sentence).to(device)
        #y_slot = torch.tensor(slot_label).to(device)
        #y_slot = utils.one_hot(y_slot).to(device)
        y_intent = torch.tensor(intent_label).to(device)
        y_intent = utils.one_hot(y_intent, Num=len(intent_dict)).to(device)

		# Calculate compute graph
        #slot_optimizer.zero_grad()
        intent_optimizer.zero_grad()
		
        #hs = slot_model.enc(x)
        #slot_model.share_memory = hs.clone()

        hi = intent_model.enc(x)
        #intent_model.share_memory = hi.clone()
		
        '''
        slot_logits = slot_model.dec(hs, intent_model.share_memory.detach())
        log_slot_logits = utils.masked_log_softmax(slot_logits, mask, dim=-1)
        slot_loss = -1.0*torch.sum(y_slot*log_slot_logits)
        slot_loss_history.append(slot_loss.item())
        slot_loss.backward()
        torch.nn.utils.clip_grad_norm_(slot_model.parameters(), 5.0)
        slot_optimizer.step()
        '''

        # Asynchronous training
        intent_logits = intent_model.dec(hi)
        log_intent_logits = F.log_softmax(intent_logits, dim=-1)
        intent_loss = -1.0*torch.sum(y_intent*log_intent_logits)
        intent_loss_history.append(intent_loss.item())
        intent_loss.backward()
        torch.nn.utils.clip_grad_norm_(intent_model.parameters(), 5.0)
        intent_optimizer.step()

        res_test = torch.argmax(log_intent_logits, dim=-1)
        res_test_list = res_test.cpu().numpy()
        com = np.array(res_test_list - intent_label)
        cnt_array = np.where(com, 0, 1)
        correct_num += np.sum(cnt_array)
        
		# Log
        if batch_index % 100 == 0 and batch_index > 0:
            print('Intent loss: {:.4f}'.format(sum(intent_loss_history[-100:])/100.0))
    print('Train Intent Acc: {:.4f}'.format(100.0 * correct_num / total_test))

    # Evaluation 
    total_test = len(test_data)
    correct_num = 0
    TP, FP, FN = 0, 0, 0
    for batch_index, data_test in enumerate(utils.get_batch(test_data, batch_size=16)):
        sentence_test, real_len_test, intent_label_test = data_test
        # print(sentence[0].shape, real_len.shape, slot_label.shape)
        x_test = torch.tensor(sentence_test).to(device)

        #mask_test = utils.make_mask(real_len_test, batch=1).to(device)
        # Slot model generate hs_test and intent model generate hi_test
        #hs_test = slot_model.enc(x_test)
        hi_test = intent_model.enc(x_test)

        # Slot
        #slot_logits_test = slot_model.dec(hs_test, hi_test)
        #log_slot_logits_test = utils.masked_log_softmax(slot_logits_test, mask_test, dim=-1)
        #slot_pred_test = torch.argmax(log_slot_logits_test, dim=-1)
        # Intent
        intent_logits_test = intent_model.dec(hi_test)
        log_intent_logits_test = F.log_softmax(intent_logits_test, dim=-1)
        res_test = torch.argmax(log_intent_logits_test, dim=-1)

        res_test_list = res_test.cpu().numpy()
        com = np.array(res_test_list - intent_label_test)
        cnt_array = np.where(com, 0, 1)
        correct_num += np.sum(cnt_array)
        # if res_test.item() == intent_label_test[0]:
        #    correct_num += 1
        if correct_num > best_correct_num:
            best_correct_num = correct_num
            best_epoch = epoch
			# Save and load the entire model.
            torch.save(intent_model, 'model_intent_best.ckpt')
    print('Best Intent Acc: {:.4f} at Epoch: [{}]'.format(100.0*best_correct_num/total_test, best_epoch+1))

train_end=time.time()
print('\nTrain time: {:.6f} seconds'.format(train_end-train_start))
test_start=time.time()
correct_num=0
total_test = len(train_data)
for batch_index, data in enumerate(utils.get_batch(train_data, batch_size=64)):
    sentence_test, real_len_test, intent_label_test = data
    # print(sentence[0].shape, real_len.shape, slot_label.shape)
    x_test = torch.tensor(sentence_test).to(device)

    # mask_test = utils.make_mask(real_len_test, batch=1).to(device)
    # Slot model generate hs_test and intent model generate hi_test
    # hs_test = slot_model.enc(x_test)
    hi_test = intent_model.enc(x_test)

    # Slot
    # slot_logits_test = slot_model.dec(hs_test, hi_test)
    # log_slot_logits_test = utils.masked_log_softmax(slot_logits_test, mask_test, dim=-1)
    # slot_pred_test = torch.argmax(log_slot_logits_test, dim=-1)
    # Intent
    intent_logits_test = intent_model.dec(hi_test)
    log_intent_logits_test = F.log_softmax(intent_logits_test, dim=-1)
    res_test = torch.argmax(log_intent_logits_test, dim=-1)

    #res_test_list = res_test.cpu().numpy()
    #com = np.array(res_test_list - intent_label_test)
    #cnt_array = np.where(com, 0, 1)
    #correct_num += np.sum(cnt_array)
    if res_test.item() == intent_label_test[0]:
        correct_num += 1
print('Training Acc: {:.4f}'.format(100.0 * correct_num / total_test))
test_end=time.time()
print('Test time: {:.6f} seconds'.format(test_end-test_start))



