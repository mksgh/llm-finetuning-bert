import os
import datasets
import json
import numpy as np
import evaluate
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer

# load dataset
conll2003 = datasets.load_dataset("lhoestq/conll2003")

# list of NER tags
ner_tags_list=conll2003["train"].features["ner_tags"].feature.names

# load model specific tokenizer
model_to_load = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_to_load)


def tokenize_and_align_labels(input_sentence, label_all_tokens=True):

    # tokeinze ids
    tokenized_inputs = tokenizer(
        input_sentence["tokens"], truncation=True, is_split_into_words=True
    )
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
tokenized_and_nermapped_datasets = conll2003.map(
    tokenize_and_align_labels, batched=True
)

# Instantiating bert-based-uncased model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=9
)

# loading evaluation metric

""" info:  seqeval is a Python framework for sequence labeling evaluation. seqeval can evaluate 
the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, 
semantic role labeling and so on."""

metric = evaluate.load("seqeval")

# defining arguments for the trainer class
args = TrainingArguments(
    "test-ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    report_to="none",  # Disable wandb logging
    save_strategy="steps",
    save_steps=0.9,
    resume_from_checkpoint=True
)

# einitializing data collector to pass data in batches to trainer
data_collator = DataCollatorForTokenClassification(tokenizer)


# defining function to compute metric
def compute_metrics(eval_preds):
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [
            ner_tags_list[eval_preds]
            for (eval_preds, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [ner_tags_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    results = metric.compute(predictions=predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# initilizing trainer
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_and_nermapped_datasets["train"],
    eval_dataset=tokenized_and_nermapped_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# training model - fine tuning for NER tags
trainer.train()

# saving model 
model.save_pretrained("./saved-model/ner_model")


# updating the model config file
model_config=json.load(open("./saved-model/ner_model/config.json"))

id2label = {str(i): label for i,label in enumerate(ner_tags_list)}
label2id = {label: str(i) for i,label in enumerate(ner_tags_list)}

model_config["id2label"]=id2label
model_config["label2id"] = label2id

json.dump(model_config,open("./saved-model/ner_model/config.json","w"))