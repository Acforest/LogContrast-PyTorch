import os
import torch
import random
import numpy as np


def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def word_insert(sent: str, word: str) -> list:
    sent_ls = sent.split()
    sent_ls.insert(np.random.randint(0, len(sent_ls) + 1), word)
    return sent_ls


def word_drop(sent: str) -> list:
    sent_ls = sent.split()
    if len(sent_ls) > 1:
        sent_ls.pop(np.random.randint(0, len(sent_ls)))
    return sent_ls


def word_replace(sent: str, word: str) -> list:
    sent_ls = sent.split()
    idx = np.random.randint(0, len(sent_ls))
    sent_ls[idx] = word
    return sent_ls


def word_shuffle(sent: str) -> list:
    sent_ls = sent.split()
    left = np.random.randint(0, len(sent_ls) - 1)
    right = np.random.randint(left + 1, len(sent_ls))
    sent_shuffle = sent_ls[left:right]
    random.shuffle(sent_shuffle)
    return sent_ls[:left] + sent_shuffle + sent_ls[right:]


def evolution(sent: str, tokenizer) -> list:
    op = np.random.randint(0, 3)
    if op == 0:
        word_id = np.random.randint(10000, 25000)
        word = ''.join(tokenizer.decode(word_id).split())
        sent_ls = word_insert(sent, word)
    elif op == 1:
        sent_ls = word_drop(sent)
    elif op == 2:
        word_id = np.random.randint(10000, 25000)
        word = ''.join(tokenizer.decode(word_id).split())
        sent_ls = word_replace(sent, word)
    else:
        sent_ls = word_shuffle(sent)

    return sent_ls
