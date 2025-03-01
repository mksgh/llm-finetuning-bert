import os
import datasets
import numpy as np
import evaluate
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification

# load dataset
conll2003 = datasets.load_dataset("conll2003", trust_remote_code=True)

# load model specific tokenizer
model_to_load = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_to_load)


def tokenize_and_align_labels(input_sentence, label_all_tokens=True):

    #tokeinze ids
    tokenized_inputs = tokenizer(input_sentence["tokens"], truncation=True, is_split_into_words=True)
    labels = []


    for i, label in enumerate(input_sentence["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token.

        previous_word_idx = None
        label_ids = []
        # Special tokens like `` and `<\s>` are originally mapped to None
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids:
            if word_idx is None:
                # set –100 as the label for these special tokens
                label_ids.append(-100)

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token
                label_ids.append(label[word_idx])
            else:
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                # mask the subword representations after the first subword

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# map enitire data with NER tags
tokenized_and_nermapped_datasets = conll2003.map(tokenize_and_align_labels, batched=True)

# Instantiating bert-based-uncased model
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased",num_labels=9)

# loading evaluation metric

''' info:  seqeval is a Python framework for sequence labeling evaluation. seqeval can evaluate 
the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, 
semantic role labeling and so on.'''

metric = evaluate.load("seqeval")

