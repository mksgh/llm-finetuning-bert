import os
import datasets
import numpy as np
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification

# load dataset
conll2003 = datasets.load_dataset("conll2003", trust_remote_code=True)