from transformers import pipeline
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification

# load model specific tokenizer
model_to_load = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_to_load)

model_fine_tuned = AutoModelForTokenClassification.from_pretrained("./saved-model/ner_model/")

nlp_pipeline=pipeline("ner",model=model_fine_tuned,tokenizer=tokenizer)

test1 = "James is a actor with good acting skills."

nlp_pipeline(test1)