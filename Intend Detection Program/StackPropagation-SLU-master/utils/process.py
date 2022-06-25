"""
@Author		:           Lee, Qin
@StartTime	:           2018/08/13
@Filename	:           process.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/05/07
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os
import time
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

# Utils functions copied from Slot-gated model, origin url:
# 	https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py
from utils import miulab


class Processor(object):

    def __init__(self, dataset, model, batch_size):
        self.__dataset = dataset
        self.__model = model
        self.__batch_size = batch_size

        if torch.cuda.is_available():
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.__criterion = nn.NLLLoss()
        self.__optimizer = optim.Adam(
            self.__model.parameters(), lr=self.__dataset.learning_rate,
            weight_decay=self.__dataset.l2_penalty
        )

    def train(self):
        #best_dev_slot = 0.0
        best_dev_intent = 0.0
        best_dev_sent = 0.0

        dataloader = self.__dataset.batch_delivery('train')
        for epoch in range(0, self.__dataset.num_epoch):
            total_intent_loss = 0.0

            time_start = time.time()
            self.__model.train()

            for text_batch, intent_batch in tqdm(dataloader, ncols=50):
                padded_text, [sorted_intent], seq_lens, _ = self.__dataset.add_padding(
                    text_batch, [(intent_batch, False)]
                )
                sorted_intent = [item * num for item, num in zip(sorted_intent, seq_lens)]
                sorted_intent = list(Evaluator.expand_list(sorted_intent))

                text_var = Variable(torch.LongTensor(padded_text))
                intent_var = Variable(torch.LongTensor(sorted_intent))

                if torch.cuda.is_available():
                    text_var = text_var.cuda()
                    #slot_var = slot_var.cuda()
                    intent_var = intent_var.cuda()

                random_intent = random.random()
                if random_intent < self.__dataset.intent_forcing_rate:
                    intent_out = self.__model(
                        text_var, seq_lens, forced_intent=intent_var
                    )
                else:
                    intent_out = self.__model(text_var, seq_lens)

                #slot_loss = self.__criterion(slot_out, slot_var)
                intent_loss = self.__criterion(intent_out, intent_var)
                batch_loss = intent_loss

                self.__optimizer.zero_grad()
                batch_loss.backward()
                self.__optimizer.step()

                try:
                    #total_slot_loss += slot_loss.cpu().item()
                    total_intent_loss += intent_loss.cpu().item()
                except AttributeError:
                    #total_slot_loss += slot_loss.cpu().data.numpy()[0]
                    total_intent_loss += intent_loss.cpu().data.numpy()[0]

            time_con = time.time() - time_start
            print('[Epoch {:2d}]: The total intent data is {:2.6f}, cost ''about {:2.6} seconds.'.format(epoch, total_intent_loss, time_con))

            change, time_start = False, time.time()
            dev_acc, dev_sent_acc = self.estimate(if_dev=True, test_batch=self.__batch_size)

            if dev_acc > best_dev_intent or dev_sent_acc > best_dev_sent:
                test_acc, test_sent_acc = self.estimate(if_dev=False, test_batch=self.__batch_size)

                if dev_acc > best_dev_intent:
                    best_dev_intent = dev_acc
                if dev_sent_acc > best_dev_sent:
                    best_dev_sent = dev_sent_acc

                print('\nTest result: intent acc score: {:.6f}, semantic ''accuracy score: {:.6f}'.format(test_acc, test_sent_acc))

                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                torch.save(self.__model, os.path.join(model_save_dir, "model.pkl"))
                torch.save(self.__dataset, os.path.join(model_save_dir, 'dataset.pkl'))

                time_con = time.time() - time_start
                print('[Epoch {:2d}]: In validation process ' \
                      'the intent acc is {:2.6f}, the semantic acc is {:.2f}, cost about ' \
                      '{:2.6f} seconds.\n'.format(epoch,  dev_acc, dev_sent_acc, time_con))

        start = time.time()
        pred_intent, real_intent, _ = self.prediction_out(
            self.__model, self.__dataset,  1
        )
        end = time.time()
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        print('Test time: {:.6f} seconds, the intent acc is {:2.6f},'.format(end - start, intent_acc))

    def estimate(self, if_dev, test_batch=100):
        """
        Estimate the performance of model on dev or test dataset.
        """

        if if_dev:
            pred_intent, real_intent, _ = self.prediction(
                self.__model, self.__dataset, "dev", test_batch
            )
        else:
            pred_intent, real_intent, _ = self.prediction(
                self.__model, self.__dataset, "test", test_batch
            )

        #slot_f1_socre = miulab.computeF1Score(pred_slot, real_slot)[0]
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_intent, real_intent)

        return intent_acc,sent_acc

    @staticmethod
    def validate(model_path, dataset_path, batch_size):
        """
        validation will write mistaken samples to files and make scores.
        """

        model = torch.load(model_path)
        dataset = torch.load(dataset_path)

        # Get the sentence list in test dataset.
        sent_list = dataset.test_sentence

        exp_pred_intent, real_intent, pred_intent = Processor.prediction(
            model, dataset, "test", batch_size
        )

        # To make sure the directory for save error prediction.
        mistake_dir = os.path.join(dataset.save_dir, "error")
        if not os.path.exists(mistake_dir):
            os.mkdir(mistake_dir)

        #slot_file_path = os.path.join(mistake_dir, "slot.txt")
        intent_file_path = os.path.join(mistake_dir, "intent.txt")
        both_file_path = os.path.join(mistake_dir, "both.txt")

        # Write those sample with mistaken slot prediction.

        # Write those sample with mistaken intent prediction.
        with open(intent_file_path, 'w') as fw:
            for w_list, p_intent_list, r_intent, p_intent in zip(sent_list, pred_intent, real_intent, exp_pred_intent):
                if p_intent != r_intent:
                    for w, p in zip(w_list, p_intent_list):
                        fw.write(w + '\t' + p + '\n')
                    fw.write(r_intent + '\t' + p_intent + '\n\n')

        # Write those sample both have intent and slot errors.

        #slot_f1 = miulab.computeF1Score(pred_slot, real_slot)[0]
        intent_acc = Evaluator.accuracy(exp_pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(exp_pred_intent, real_intent)

        return intent_acc, sent_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size):
        model.eval()

        if mode == "dev":
            dataloader = dataset.batch_delivery('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "train":
            dataloader = dataset.batch_delivery('train', batch_size=batch_size, shuffle=False, is_digital=False)
            mode = "test"
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        #pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []

        for text_batch, intent_batch in tqdm(dataloader, ncols=50):
            padded_text, [sorted_intent], seq_lens, sorted_index = dataset.add_padding(
                text_batch, [(intent_batch, False)], digital=False
            )
            # Because it's a visualization bug, in valid time, it doesn't matter
            # Only in test time will it need to restore
            if mode == 'test':
                #tmp_r_slot = [[] for _ in range(len(sorted_index))]
                #for i in range(len(sorted_index)):
                #    tmp_r_slot[sorted_index[i]] = sorted_slot[i]
                #sorted_slot = tmp_r_slot
                tmp_intent = [[] for _ in range(len(sorted_index))]
                for i in range(len(sorted_index)):
                    tmp_intent[sorted_index[i]] = sorted_intent[i]
                sorted_intent = tmp_intent
            
            #real_slot.extend(sorted_slot)
            real_intent.extend(list(Evaluator.expand_list(sorted_intent)))

            digit_text = dataset.word_alphabet.get_index(padded_text)
            var_text = Variable(torch.LongTensor(digit_text))

            if torch.cuda.is_available():
                var_text = var_text.cuda()

            intent_idx = model(var_text, seq_lens, n_predicts=1)
            #nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]

            nested_intent = Evaluator.nested_list([list(Evaluator.expand_list(intent_idx))], seq_lens)[0]
            
            if mode == 'test':
                tmp_intent = [[] for _ in range(len(sorted_index))]
                for i in range(len(sorted_index)):
                    tmp_intent[sorted_index[i]] = nested_intent[i]
                nested_intent = tmp_intent
            
            pred_intent.extend(dataset.intent_alphabet.get_instance(nested_intent))

        exp_pred_intent = Evaluator.max_freq_predict(pred_intent)
        return exp_pred_intent, real_intent, pred_intent

    def prediction_out(self, model, dataset, batch_size=1):
        model.eval()

        dataloader = dataset.batch_delivery('train', batch_size=batch_size, shuffle=False, is_digital=False)

        pred_intent, real_intent = [], []

        for text_batch, intent_batch in tqdm(dataloader, ncols=50):
            padded_text, [sorted_intent], seq_lens, sorted_index = dataset.add_padding(
                text_batch, [(intent_batch, False)], digital=False
            )
            # real_slot.extend(sorted_slot)
            real_intent.extend(list(Evaluator.expand_list(sorted_intent)))

            digit_text = dataset.word_alphabet.get_index(padded_text)
            var_text = Variable(torch.LongTensor(digit_text))

            if torch.cuda.is_available():
                var_text = var_text.cuda()

            intent_idx = model(var_text, seq_lens, n_predicts=1)
            # nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]

            nested_intent = Evaluator.nested_list([list(Evaluator.expand_list(intent_idx))], seq_lens)[0]

            pred_intent.extend(dataset.intent_alphabet.get_instance(nested_intent))

        exp_pred_intent = Evaluator.max_freq_predict(pred_intent)
        return exp_pred_intent, real_intent, pred_intent


class Evaluator(object):

    @staticmethod
    def semantic_acc(pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """

        total_count, correct_count = 0.0, 0.0
        for p_intent, r_intent in zip(pred_intent, real_intent):

            if p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def f1_score(pred_list, real_list):
        """
        Get F1 score measured by predictions and ground-trues.
        """

        tp, fp, fn = 0.0, 0.0, 0.0
        for i in range(len(pred_list)):
            seg = set()
            result = [elem.strip() for elem in pred_list[i]]
            target = [elem.strip() for elem in real_list[i]]

            j = 0
            while j < len(target):
                cur = target[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(target):
                        str_ = target[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    seg.add((cur, j, k - 1))
                    j = k - 1
                j = j + 1

            tp_ = 0
            j = 0
            while j < len(result):
                cur = result[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(result):
                        str_ = result[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    if (cur, j, k - 1) in seg:
                        tp_ += 1
                    else:
                        fp += 1
                    j = k - 1
                j = j + 1

            fn += len(seg) - tp_
            tp += tp_

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        return 2 * p * r / (p + r) if p + r != 0 else 0

    """
    Max frequency prediction. 
    """

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def exp_decay_predict(sample, decay_rate=0.8):
        predict = []
        for items in sample:
            item_dict = {}
            curr_weight = 1.0
            for item in items[::-1]:
                item_dict[item] = item_dict.get(item, 0) + curr_weight
                curr_weight *= decay_rate
            predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
