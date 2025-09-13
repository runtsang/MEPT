import os
import sys
if "prompt_expert" not in os.getcwd():
    os.chdir("prompt_expert")
sys.path.append(os.getcwd())

import random
from tqdm import tqdm
import pickle
import ast

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.data.datasets import *
from datasets import load_dataset
from collections import defaultdict
import copy


def construct_dev_from_train_t2t(train_list, num_labels):
    if num_labels == 1:
        border = len(train_list) * 8 // 9
        val_list = train_list[border:]
        actual_train_list = train_list[:border]
    else:
        actual_train_list = []
        val_list = []
        label_dict = defaultdict(list)
        for input_text, label_text in train_list:
            label_dict[label_text].append(input_text)
        
        for label_text in label_dict:
            inputs = label_dict[label_text]
            n_inputs = len(inputs)
            border = n_inputs * 9 // 10
            actual_train_list.extend([(i, label_text) for i in inputs[:border]])
            val_list.extend([(i, label_text) for i in inputs[border:]])

    return actual_train_list, val_list


def boolq(text_to_text):
    path = "data/superglue/boolq"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        # https://huggingface.co/datasets/stjokerli/TextToText_boolq/viewer/stjokerli--TextToText_boolq/train
        text_format = f"boolq passage: **passage** question: **question**"
        labels = ("False", "True")
        train_list = [(text_format.replace("**passage**", d['passage']).replace("**question**", d['question']), labels[d['label']]) for d in train]
        val_list = [(text_format.replace("**passage**", d['passage']).replace("**question**", d['question']), labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        text_format = f"boolq passage: **passage** question: **question**"
        labels = ("False", "True")
        train_list = [(text_format.replace("**passage**", d['passage']).replace("**question**", d['question']), d['label']) for d in train]
        val_list = [(text_format.replace("**passage**", d['passage']).replace("**question**", d['question']), d['label']) for d in val]
        num_labels = len(labels)
        # train_list = [((d['question'], d['passage']), d['label']) for d in train]
        # val_list = [((d['question'], d['passage']), d['label']) for d in val]
        # num_labels = max(train['label']) + 1

    return train_list, val_list, num_labels


def cb(text_to_text):
    path = "data/superglue/cb"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        text_format = f"cb hyopthesis: **hypothesis**. premise: **premise**"
        labels = ('entailment', 'contradiction', 'neutral')
        train_list = [(text_format.replace("**hypothesis**", d['hypothesis']).replace("**premise**", d['premise']), labels[d['label']]) for d in train]
        val_list =  [(text_format.replace("**hypothesis**", d['hypothesis']).replace("**premise**", d['premise']), labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        text_format = f"cb hyopthesis: **hypothesis**. premise: **premise**"
        labels = ('entailment', 'contradiction', 'neutral')
        train_list = [(text_format.replace("**hypothesis**", d['hypothesis']).replace("**premise**", d['premise']), d['label']) for d in train]
        val_list =  [(text_format.replace("**hypothesis**", d['hypothesis']).replace("**premise**", d['premise']), d['label']) for d in val]
        num_labels = len(labels)
        # train_list = [((d['premise'], d['hypothesis']), d['label']) for d in train]
        # val_list =  [((d['premise'], d['hypothesis']), d['label']) for d in val]
        # num_labels = max(train['label']) + 1
    return train_list, val_list, num_labels

def copa(text_to_text):
    path = "data/superglue/copa"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        text_format = "copa choice1: **choice1** choice2: **choice2** premise: **premise** question: **question**"
        labels = ('choice1', 'choice2')
        train_list = [(f"choice1: {d['choice1']} choice2: {d['choice2']} premise: {d['premise']} question: {d['question']}", labels[d['label']]) for d in train]
        val_list = [(f"choice1: {d['choice1']} choice2: {d['choice2']} premise: {d['premise']} question: {d['question']}", labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        text_format = "copa choice1: **choice1** choice2: **choice2** premise: **premise** question: **question**"
        labels = ('choice1', 'choice2')
        train_list = [(f"choice1: {d['choice1']} choice2: {d['choice2']} premise: {d['premise']} question: {d['question']}", d['label']) for d in train]
        val_list = [(f"choice1: {d['choice1']} choice2: {d['choice2']} premise: {d['premise']} question: {d['question']}", d['label']) for d in val]
        num_labels = len(labels)
        # train_list, val_list = [], []
        # for d in train:
        #     joiner = 'beacuse' if d['question'] == 'cause' else 'so'
        #     text_a = f"{d['premise']} {joiner}"
        #     train_list.append((((text_a, d['choice1']), (text_a, d['choice2'])), d['label']))
        # for d in val:
        #     joiner = 'beacuse' if d['question'] == 'cause' else 'so'
        #     text_a = f"{d['premise']} {joiner}"
        #     val_list.append((((text_a, d['choice1']), (text_a, d['choice2'])), d['label']))
        # num_labels = max(train['label']) + 1
    return train_list, val_list, num_labels


def multirc(text_to_text):
    path = "data/superglue/multirc"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        text_format = "multirc question: **question** answer: **answer**. paragraph: **paragraph**"
        labels = ["False", "True"]
        train_list = [(f"question: {d['question']} answer: {d['answer']}. paragraph: {d['paragraph']}", labels[d['label']]) for d in train]
        val_list = [(f"question: {d['question']} answer: {d['answer']}. paragraph: {d['paragraph']}", labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        text_format = "multirc question: **question** answer: **answer**. paragraph: **paragraph**"
        labels = ["False", "True"]
        train_list = [(f"question: {d['question']} answer: {d['answer']}. paragraph: {d['paragraph']}", d['label']) for d in train]
        val_list = [(f"question: {d['question']} answer: {d['answer']}. paragraph: {d['paragraph']}", d['label']) for d in val]
        num_labels = len(labels)
        # train_list = [((d['paragraph'], f"{d['question']} {d['answer']}"), d['label']) for d in train]
        # val_list = [((d['paragraph'], f"{d['question']} {d['answer']}"), d['label']) for d in val]
        # num_labels = max(train['label']) + 1

    return train_list, val_list, num_labels


def rte(text_to_text):
    path = "data/superglue/rte"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        # text_format = "rte sentence1: **premise** sentence2: **hypothesis**"
        text_format = "rte sentence1: **premise** sentence2: **hypothesis**"
        labels = ("entailment", "not_entailment")
        train_list = [(text_format.replace("**premise**", d['premise']).replace("**hypothesis**", d['hypothesis']), labels[d['label']]) for d in train]
        val_list = [(text_format.replace("**premise**", d['premise']).replace("**hypothesis**", d['hypothesis']), labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        text_format = "rte sentence1: **premise** sentence2: **hypothesis**"
        labels = ("entailment", "not_entailment")
        train_list = [(text_format.replace("**premise**", d['premise']).replace("**hypothesis**", d['hypothesis']), d['label']) for d in train]
        val_list = [(text_format.replace("**premise**", d['premise']).replace("**hypothesis**", d['hypothesis']), d['label']) for d in val]
        num_labels = len(labels)
        # train_list = [((d['premise'], d['hypothesis']), d['label']) for d in train]
        # val_list =  [((d['premise'], d['hypothesis']), d['label']) for d in val]
        # num_labels = max(train['label']) + 1
    
    return train_list, val_list, num_labels

def wic(text_to_text):
    path = "data/superglue/wic"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        text_format = "wic sentence1: **sentence1** sentence2: **sentence2** word: **word**"
        labels = ('False', 'True')
        train_list = [(text_format.replace("**sentence1**", d['sentence1']).replace("**sentence2**", d['sentence2']).replace("**word**", d['word']), labels[d['label']]) for d in train]
        val_list = [(text_format.replace("**sentence1**", d['sentence1']).replace("**sentence2**", d['sentence2']).replace("**word**", d['word']), labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        text_format = "wic sentence1: **sentence1** sentence2: **sentence2** word: **word**"
        labels = ('False', 'True')
        train_list = [(text_format.replace("**sentence1**", d['sentence1']).replace("**sentence2**", d['sentence2']).replace("**word**", d['word']), d['label']) for d in train]
        val_list = [(text_format.replace("**sentence1**", d['sentence1']).replace("**sentence2**", d['sentence2']).replace("**word**", d['word']), d['label']) for d in val]
        num_labels = len(labels)
        # train_list = [(f"{d['sentence1']} {d['sentence2']} Does {d['word']} have the same meaning in both sentences?", d['label']) for d in train]
        # val_list = [(f"{d['sentence1']} {d['sentence2']} Does {d['word']} have the same meaning in both sentences?", d['label']) for d in val]
        # num_labels = max(train['label']) + 1
    return train_list, val_list, num_labels

# https://github.com/google-research/text-to-text-transfer-transformer/blob/fce4b1a7fcca858482ac60579cf8b3332c594a55/t5/data/preprocessors.py#L1286
def _wsc_inputs(example):
    """
    Given an example from SuperGLUE WSC, compute the 'inputs' value as a fill-in-
    the-blank. We replace the pronoun with 'X'.

    The output will look like a fill in the blank with the pronoun blanked out.
    For example, 
      'Mitchell asked Tom if he could lend some money.'
    becomes
      'Mitchell asked Tom if X could lend some money.'

    Some special cases are handled explicitly per the original code.

    Args:
        example (dict): Must contain:
          - 'text': str
          - 'span2_text': str (the pronoun)
          - 'span2_index': int (pronoun index in the text)

    Returns:
        str: The text with the pronoun replaced by 'X', or a special-case string.
    """
    # print(example)
    text = example['text']
    span2_text = example["span2_text"]
    span2_index = example["span2_index"]

    # Check for special-case strings:
    # 1) "The boy continued to whip the pony..."
    special_case1 = (
        'The boy continued to whip the pony , and eventually the pony threw him over. '
        'John laughed out quite loud. "Good for him," he said. '
    )
    if text == special_case1:
        return (
            'The boy continued to whip the pony , and eventually the pony threw '
            'him over. John laughed out quite loud. "Good for X ," he said.'
        )

    # 2) "When they had eventually calmed down..."
    special_case2 = (
        'When they had eventually calmed down a bit , and had gotten home, '
        'Mr. Farley put the magic pebble in an iron safe . Some day they might want '
        'to use it , but really for now, what more could they wish for?'
    )
    if text == special_case2:
        return (
            'When they had eventually calmed down a bit , and had gotten home, '
            'Mr. Farley put the magic pebble in an iron safe . Some day they might '
            'want to use X , but really for now, what more could they wish for?'
        )

    # Normal (non-special) case:
    words = text.split()
    # Basic checks (the original code uses tf.assert_* but we'll do simple Python checks):
    if not (0 < span2_index < len(words)):
        # If your data has guaranteed consistency, you may prefer to raise an error.
        # Otherwise, we can just return the text as-is.
        return text

    # Check that words[span2_index] matches the pronoun we expect:
    if words[span2_index] != span2_text:
        # If there's a mismatch, either raise an error or proceed carefully.
        pass

    # Replace the pronoun with 'X':
    return " ".join([
        " ".join(words[:span2_index]), 
        "X", 
        " ".join(words[span2_index + 1:])
    ]).strip()

def wsc_simple(
    examples, 
    label="wsc:",
    correct_referent_only=False
):
    """
    Converts SuperGLUE WSC examples to a simple text-to-text format.

    For each example in `examples`, we transform:
        text: "Mitchell asked Tom if he could lend some money."
        span1_text: "Tom"
        span2_text: "he"
        span2_index: 4
    into something like:
        inputs: "wsc: Mitchell asked Tom if *he* could lend some money."
        targets: "Tom"

    If `correct_referent_only` is True, we will only keep examples for which
    x['label'] == 1 (i.e., the pronoun does indeed refer to span1_text).

    Args:
        examples (list of dict): Each dict must have:
          - 'text': str
          - 'span1_text': str (the candidate referent)
          - 'span2_text': str (the pronoun)
          - 'span2_index': int (the pronoun’s index in 'text')
          - 'idx': anything you want to track (int or dict)
          - optionally 'label': int or bool (1 => correct, 0 => incorrect)
        label (str): A label to prepend to the 'inputs'.
        correct_referent_only (bool): Whether to filter out non-correct referents.

    Returns:
        list of dicts: Each dict has:
          - 'inputs'
          - 'targets'
          - 'label'
          - 'idx'
    """

    output = []

    for x in examples:
        # If correct_referent_only is True, we skip any example with label != 1
        # (assuming 1 => correct pronoun reference, 0 or None => incorrect).
        if correct_referent_only and x.get("label", 0) != 1:
            continue

        # Create a shallow copy to avoid mutating the original
        ex = copy.copy(x)
        
        # 1) Create the fill-in version by calling _wsc_inputs
        fill_in_version = _wsc_inputs(ex)  # text with pronoun replaced by 'X'
        
        # 2) Insert the pronoun back in place of 'X', but with asterisks.
        #    e.g. "X" -> "*he*"
        #    We'll do a naive string replacement for ' X ' -> ' *he* '
        #    If 'X' might appear at the beginning or end, handle that carefully.
        #    For simplicity, we'll do a global replacement. 
        #    (In practice, if your data can have the letter "X" appear, 
        #     you'd want a safer approach.)
        starred_pronoun = f"*{ex['span2_text']}*"
        # We'll replace ' X ' or ' X,' or ' X.' etc. 
        # A simple approach is to split, replace, re-join:
        tokens = fill_in_version.split()
        for i, tok in enumerate(tokens):
            if tok == 'X':
                tokens[i] = starred_pronoun
        fill_in_version_starred = " ".join(tokens)

        # 3) Build the final `inputs`
        #    e.g. "wsc: Mitchell asked Tom if *he* could lend some money."
        final_inputs = f"{label} {fill_in_version_starred}"

        # 4) The target is always 'span1_text'
        target = x["span1_text"]

        # 5) Construct the final dictionary
        result = {
            "inputs": final_inputs,
            "targets": target,
            "label": x.get("label", 0),   # or None if not provided
            "idx": x["idx"]
        }

        output.append(result)

    return output


def wsc_simple(
    examples, 
    label="wsc:",
    correct_referent_only=False
):
    """
    Converts SuperGLUE WSC examples to a simple text-to-text format.

    For each example in `examples`, we transform:
        text: "Mitchell asked Tom if he could lend some money."
        span1_text: "Tom"
        span2_text: "he"
        span2_index: 4
    into something like:
        inputs: "wsc: Mitchell asked Tom if *he* could lend some money."
        targets: "Tom"

    If `correct_referent_only` is True, we will only keep examples for which
    x['label'] == 1 (i.e., the pronoun does indeed refer to span1_text).

    Args:
        examples (list of dict): Each dict must have:
          - 'text': str
          - 'span1_text': str (the candidate referent)
          - 'span2_text': str (the pronoun)
          - 'span2_index': int (the pronoun’s index in 'text')
          - 'idx': anything you want to track (int or dict)
          - optionally 'label': int or bool (1 => correct, 0 => incorrect)
        label (str): A label to prepend to the 'inputs'.
        correct_referent_only (bool): Whether to filter out non-correct referents.

    Returns:
        list of dicts: Each dict has:
          - 'inputs'
          - 'targets'
          - 'label'
          - 'idx'
    """

    output = []

    for x in examples:
        # If correct_referent_only is True, we skip any example with label != 1
        # (assuming 1 => correct pronoun reference, 0 or None => incorrect).
        if correct_referent_only and x.get("label", 0) != 1:
            continue

        # Create a shallow copy to avoid mutating the original
        ex = copy.copy(x)
        
        # 1) Create the fill-in version by calling _wsc_inputs
        fill_in_version = _wsc_inputs(ex)  # text with pronoun replaced by 'X'
        
        # 2) Insert the pronoun back in place of 'X', but with asterisks.
        #    e.g. "X" -> "*he*"
        #    We'll do a naive string replacement for ' X ' -> ' *he* '
        #    If 'X' might appear at the beginning or end, handle that carefully.
        #    For simplicity, we'll do a global replacement. 
        #    (In practice, if your data can have the letter "X" appear, 
        #     you'd want a safer approach.)
        starred_pronoun = f"*{ex['span2_text']}*"
        # We'll replace ' X ' or ' X,' or ' X.' etc. 
        # A simple approach is to split, replace, re-join:
        tokens = fill_in_version.split()
        for i, tok in enumerate(tokens):
            if tok == 'X':
                tokens[i] = starred_pronoun
        fill_in_version_starred = " ".join(tokens)

        # 3) Build the final `inputs`
        #    e.g. "wsc: Mitchell asked Tom if *he* could lend some money."
        final_inputs = f"{label} {fill_in_version_starred}"

        # 4) The target is always 'span1_text'
        target = x["span1_text"]

        # 5) Construct the final dictionary
        result = {
            "inputs": final_inputs,
            "targets": target,
            "label": x.get("label", 0),   # or None if not provided
            "idx": x["idx"]
        }

        output.append(result)

    return output

def wsc(text_to_text):
    path = "data/superglue/wsc"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        # filter_train = wsc_simple(train, correct_referent_only=True)
        # filter_val = wsc_simple(val, correct_referent_only=True)

        # train_list = [(d['inputs'], d['targets']) for d in filter_train]
        # val_list = [(d['inputs'], d['targets']) for d in filter_val]

        text_format = "wsc text: **text** span1_text: **span1_text** span2_text: **span2_text**"
        labels = ('Different', 'Same')
        train_list = [(text_format.replace("**text**", d['text']).replace("**span1_text**", d['span1_text']).replace("**span2_text**", d['span2_text']), labels[d['label']]) for d in train]
        val_list = [(text_format.replace("**text**", d['text']).replace("**span1_text**", d['span1_text']).replace("**span2_text**", d['span2_text']), labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        # https://github.com/AmanDaVinci/lifelong-learning-limitations/blob/555e4b65eab285f91492451c56fa668fe3d71d4b/src/datastream.py#L169
        train_list = [(f"{d['text']} {d['span1_text']} Does {d['span2_text']} refer to {d['span1_text']} in {d['text']}?", d['label']) for d in train]
        val_list = [(f"{d['text']} {d['span1_text']} Does {d['span2_text']} refer to {d['span1_text']} in {d['text']}?", d['label']) for d in val]
        num_labels = max(train['label']) + 1

    return train_list, val_list, num_labels

import re
# https://github.com/google-research/text-to-text-transfer-transformer/blob/fce4b1a7fcca858482ac60579cf8b3332c594a55/t5/data/preprocessors.py#L917
def record_ex(examples):
    """
    Convert ReCoRD examples into multiple text2text examples—one per answer.

    Each item in `examples` should be a dictionary with the following keys:
      - 'passage': str
      - 'query': str (contains '@placeholder')
      - 'entities': List[str]
      - 'answers': List[str] (any can be correct)
      - 'idx': dict with at least 'passage' and 'query' keys (optional)

    This function returns a list of dictionaries. Each dictionary has:
      - 'inputs': "record query: ... entities: ... passage: ..."
      - 'targets': a single correct answer (or '<unk>' if none)
      - 'answers': the entire list of possible answers (for eval)
      - 'idx/passage', 'idx/query': optional indices from the original data
    """
    output = []

    for x in examples:
        # 1) Clean up the passage by replicating the regex replacements:
        passage = x["passage"]
        passage = re.sub(r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
        passage = re.sub(r'\n@highlight\n', '. ', passage)

        # 2) Build the string for inputs
        query_str = x["query"]
        entities_str = ", ".join(x["entities"])
        inputs_str = f"record query: {query_str} entities: {entities_str} passage: {passage}"

        # 3) Create one example per answer
        answers = x.get("answers", [])
        if len(answers) == 0:
            # If no answers, produce a single example with "<unk>"
            example_dict = {
                "inputs": inputs_str,
                "targets": "<unk>",
                "answers": [],  # No valid answers
            }
            # Add optional indices
            if "idx" in x and isinstance(x["idx"], dict):
                example_dict["idx/passage"] = x["idx"].get("passage", None)
                example_dict["idx/query"]   = x["idx"].get("query", None)
            output.append(example_dict)
        else:
            # If we have answers, create an example for each one
            for ans in answers:
                example_dict = {
                    "inputs": inputs_str,
                    "targets": ans,
                    "answers": answers,  # pass entire set of possible answers
                }
                # Add optional indices
                if "idx" in x and isinstance(x["idx"], dict):
                    example_dict["idx/passage"] = x["idx"].get("passage", None)
                    example_dict["idx/query"]   = x["idx"].get("query", None)

                output.append(example_dict)

    return output

def record(text_to_text):
    path = "data/superglue/record"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    def format_text_to_text(d):
        text_format = "record passage: {passage} query: {query} entities: {entities}"
        entities_str = ', '.join(f"'{entity}'" for entity in d['entities'])
        return text_format.format(
            passage=d['passage'],
            query=d['query'],
            entities=entities_str
        )

    def format_non_text_to_text(d):
        entities_str = ', '.join(f"'{entity}'" for entity in d['entities'])
        text_format = "{passage} {query} {entities} Which entity in {entities} should @placeholder in {query} be?"
        return text_format.format(
            passage=d['passage'],
            query=d['query'],
            entities=entities_str
        )

    if text_to_text:
        filter_train = record_ex(train)
        filter_val = record_ex(val)
        train_list = [(d['inputs'], d['targets']) for d in filter_train]
        val_list = [(d['inputs'], d['targets']) for d in filter_val]
    else:
        train_list = [(format_non_text_to_text(d), d['answers'][0]) for d in train]
        val_list = [(format_non_text_to_text(d), d['answers'][0]) for d in val]

    num_labels = 0

    return train_list, val_list, num_labels

def load_and_combine_splits(data_list, text_to_text):
    """
    Load the training and validation splits from multiple datasets, and merge the train and val splits separately across datasets.
    
    Args:
    - data_list (list): List of dataset names, e.g., ['boolq', 'cb', 'copa'].
    - text_to_text (bool): Whether to load in text-to-text format.
    
    Returns:
    - combined_train (list): Merged training set.
    - combined_val (list): Merged validation set.
    """
    # Define the dataset processing function mapping
    dataset_names = data_list
    print(data_list)
    
    dataset_functions = {
        "boolq": boolq,
        "cb": cb,
        "copa": copa,
        "multirc": multirc,
        "rte": rte,
        "wic": wic,
        "wsc": wsc,
        "record": record,
    }
    
    combined_train = []
    combined_val = []
    
    for dataset_name in dataset_names:
        if dataset_name not in dataset_functions:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Supported datasets are: {list(dataset_functions.keys())}")
        
        dataset_function = dataset_functions[dataset_name]
        train_list, val_list, _ = dataset_function(text_to_text)
        
        combined_train.extend(train_list)
        combined_val.extend(val_list)
    
    print(f"Combined train dataset has {len(combined_train)} samples.")
    print(f"Combined val dataset has {len(combined_val)} samples.")

    return combined_train, combined_val, 2

def get_superglue(data_name, split, text_to_text=False):
    # ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']

    # Define a dictionary that maps data names to corresponding functions
    # data_funcs = {'boolq': boolq, 'cb': cb, 'copa': copa, 'multirc': multirc, 'rte_superglue': rte, 'wic': wic}
    # "['cb', 'copa']"
    if len(data_name)>10:
        data_list = ast.literal_eval(data_name)
    data_funcs = {'boolq': boolq, 'cb': cb, 'copa': copa, 'multirc': multirc, 'rte': rte, 'wic': wic, 'wsc': wsc, 'record': record}
    # if data_name not in data_funcs:
    #     raise ValueError(f"Invalid data_name '{data_name}'.")

    if data_name == 'semeval':
        train_list, val_list, test_list, num_labels = data_funcs[data_name](text_to_text)
    elif len(data_name)>10:
        train_list, val_list, num_labels = load_and_combine_splits(data_list, text_to_text)
        # val_list  is a list [a, b, c]
    else:
        train_list, val_list, num_labels = data_funcs[data_name](text_to_text)
        
    # test_list  is a list [a, b, c] if we combine datasets
    test_list = val_list
    train_list, val_list = construct_dev_from_train_t2t(train_list, num_labels)

    if split == "train":
        return train_list, num_labels
    elif split == "dev":
        return val_list, num_labels
    elif split == "test":
        return test_list, num_labels
    else:
        raise ValueError