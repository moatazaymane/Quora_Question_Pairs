import transformers

DEVICE = 'cuda'
EPOCHS = 3
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
MAX_LEN = 256
POOLED_OUTPUT_DIM = 768
lr = 3e-5

TRAINING_DATASET = 'train.csv'
TEST_DATASET = 'test.csv'
BERT_PATH = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case = True)
