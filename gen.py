from collections import Counter
from tqdm import tqdm
import nltk
import csv
import criteria
import numpy as np
import torch
import dataloader
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertForMaskedLM
from nltk.stem import PorterStemmer
import spacy

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_whole_word_to_feature(tokens_a, mask_position, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    #tokens_a = tokenizer.tokenize(sentence)
    #print(mask_position)

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)

    tokens.append("[SEP]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(1)

    tokens.append("[SEP]")
    input_type_ids.append(1)

    true_word = ''
    index = 0
    count = 0
    mask_position_length = len(mask_position)

    while count in range(mask_position_length):
        index = mask_position_length - 1 - count

        pos = mask_position[index]
        if index == 0:
            tokens[pos] = '[MASK]'
        else:
            del tokens[pos]
            del input_type_ids[pos]

        count += 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
    input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)
    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length


    return InputFeatures(unique_id=0,  tokens=tokens, input_ids=input_ids,input_mask=input_mask,input_type_ids=input_type_ids)


def convert_token_to_feature(tokens_a, mask_position, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    #tokens_a = tokenizer.tokenize(sentence)
    #print(mask_position)

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)

    tokens.append("[SEP]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(1)

    tokens.append("[SEP]")
    input_type_ids.append(1)

    true_word = ''
    if isinstance(mask_position,list):
        for pos in  mask_position:
            true_word = true_word + tokens[pos]
            tokens[pos] = '[MASK]'
    else:
        true_word = tokens[mask_position]
        tokens[mask_position] =  '[MASK]'


    input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
    input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0,  tokens=tokens, input_ids=input_ids,input_mask=input_mask,input_type_ids=input_type_ids)

def convert_sentence_to_token(sentence, seq_length, tokenizer):

    tokenized_text = tokenizer.tokenize(sentence.lower())

    assert len(tokenized_text) < seq_length-2

    nltk_sent = sentence.split(" ")
    #print(nltk_sent)
    #print(tokenized_text)

    position2 = []

    token_index = 0

    start_pos =  len(tokenized_text)  + 2

    pre_word = ""

    for i,word in enumerate(nltk_sent):

        if word=="n't" and pre_word[-1]=="n":
            word = "'t"

        if tokenized_text[token_index]=="\"":
            len_token = 2

        else:
            len_token = len(tokenized_text[token_index])

        if tokenized_text[token_index]==word or len_token>=len(word):
            position2.append(start_pos+token_index)
            pre_word = tokenized_text[token_index]
            token_index += 1
        else:
            new_pos = []
            new_pos.append(start_pos+token_index)

            new_word = tokenized_text[token_index]

            while new_word != word:

                token_index += 1

                new_word += tokenized_text[token_index].replace('##','')

                new_pos.append(start_pos+token_index)

                if len(new_word)==len(word):
                    break
            token_index += 1
            pre_word = new_word
            position2.append(new_pos)

    return tokenized_text, nltk_sent, position2

def substitution_generation(source_word, pre_tokens, pre_scores, ps, num_selection=10):

    cur_tokens=[]

    source_stem = ps.stem(source_word)

    assert num_selection<=len(pre_tokens)

    for i in range(len(pre_tokens)):
        token = pre_tokens[i]

        if token[0:2]=="##":
            continue

        if(token==source_word):
            continue

        token_stem = ps.stem(token)

        if(token_stem == source_stem):
            continue

        if (len(token_stem)>=3) and (token_stem[:3]==source_stem[:3]):
            continue
        cur_tokens.append(token)
        if(len(cur_tokens)==num_selection):
            break

    if(len(cur_tokens)==0):
        cur_tokens = pre_tokens[0:num_selection+1]


    assert len(cur_tokens)>0

    return cur_tokens



#stop_words_set = criteria.get_stopwords()

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForMaskedLM.from_pretrained('bert-base-uncased')

#texts, labels = dataloader.read_corpus('data/mr',csvf=False)
#data = list(zip(texts, labels))

#sim_score_window = 40
#nlp = spacy.load("en_core_web_sm")

#for sample_idx, (text, true_label) in enumerate(data):

#    rows = []
#    txts = []
#    words_perturb = []
#    with open('scores-mr.csv', 'r') as csvfile:
#        csvreader = csv.reader(csvfile)
#        fields = next(csvreader)
#        for row in csvreader:
#            rows.append(row)

#    doc = nlp(' '.join(text))
#    text = []
#    for sent in doc.sents:
#        for token in sent:
#            text.append(token.text)
#    tok_text = []

#    for item in text:
#        ap = item.find("'")
#        if ap>=0:
#            tok_text.append(item[0:ap])
#            tok_text.append("'")
#            tok_text.append(item[ap+1:len(item)])
#        else:
#            tok_text.append(item)

#    text = []
#    for item in tok_text:
#        if len(item) > 0:
#            text.append(item)

#    for wrd in rows[sample_idx]:
#        indx = wrd.find(" ")
#        iidx = wrd[1:indx-1]
#        widx = int(iidx)

#        if widx < len(text):
#            final_word = text[widx]
#            if final_word not in stop_words_set and len(final_word) > 2:
#                words_perturb.append((final_word,widx))

#    for wrd,idx in words_perturb:
#        len_text = len(text)
#        half_sim_score_window = (sim_score_window - 1) // 2
        # Output top 10 of candidates
