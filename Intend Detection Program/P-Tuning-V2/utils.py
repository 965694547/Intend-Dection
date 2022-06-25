import os
import random
import logging

import torch
import numpy as np

from transformers import BertConfig
from transformers import BertTokenizer

from model import JointBERT, P_Tuning_V2, Prompt

MODEL_CLASSES = {
    'bert_en': (BertConfig, JointBERT, BertTokenizer),
    'bert_ch': (BertConfig, JointBERT, BertTokenizer),
    'p_tuning_v2_en': (BertConfig, P_Tuning_V2, BertTokenizer),
    'p_tuning_v2_ch': (BertConfig, P_Tuning_V2, BertTokenizer),
}


MODEL_PATH_MAP = {
    'bert_en': 'bert-base-uncased',
    'bert_ch': 'bert-base-chinese',
    'p_tuning_v2_en': 'bert-base-uncased',
    'p_tuning_v2_ch': 'bert-base-chinese',
}


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    VOCAB= os.path.join(args.model_name_or_path,MODEL_PATH_MAP[args.model_type] + '-vocab.txt')
    return MODEL_CLASSES[args.model_type][2].from_pretrained(VOCAB)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels,):
    assert len(intent_preds) == len(intent_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    #slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels)

    results.update(intent_result)
    #results.update(slot_result)
    results.update(sementic_result)

    return results

def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)
    #sementic_acc = np.multiply(intent_result, slot_result).mean()
    sementic_acc = intent_result.mean()
    return {
        "sementic_frame_acc": sementic_acc
    }
