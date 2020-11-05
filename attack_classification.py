import argparse
import os
import numpy as np
import pickle
import dataloader
from train_classifier import Model
from itertools import zip_longest
import criteria
from fuzzywuzzy import fuzz
import random
import csv
import gen
from nltk.stem import PorterStemmer
from gen import InputFeatures
import math
import sys
import spacy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertForMaskedLM
csv.field_size_limit(sys.maxsize)
#import tensorflow as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
#tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
#import tensorflow as tf
#tf.disable_v2_behavior()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig
tf.compat.v1.disable_eager_execution()



class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores

def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + 100]
    sim_words, sim_values = [], []
    sim_dct = defaultdict(float)
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
        sim_dct[(src_word,sim_word)] = sim_value
    return sim_words, sim_values, sim_dct


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.
    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader

def calc_similarity(src_word,candidate_word):

    src_word = np.asarray(src_word)
    candidate_word = np.asarray(candidate_word)
    src_word = src_word / np.linalg.norm(src_word)
    candidate_word = candidate_word / np.linalg.norm(candidate_word)
    return np.dot(src_word.T,candidate_word)


def calc_window(idx, half_sim_score_window, sim_score_window, len_text):

    half_sim_score_window = (sim_score_window - 1) // 2

    if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = idx - half_sim_score_window
        text_range_max = idx + half_sim_score_window + 1
    elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = 0
        text_range_max = sim_score_window
    elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        text_range_min = len_text - sim_score_window
        text_range_max = len_text
    else:
        text_range_min = 0
        text_range_max = len_text
    if text_range_min < 0:
        text_range_min = 0
    if text_range_max > len_text:
        text_range_max = len_text

    return text_range_min, text_range_max

def attack(fuzz_val,top_k_words,qrs,wts,sample_index,text_ls, true_label,
           predictor, stop_words_set, word2idx, idx2word, cos_sim, word_embedding, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):
    rows=[]
    nlp = spacy.load('en_core_web_sm')
    masked_lang_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    changed_with=[]
    doc = nlp(' '.join(text_ls))
    text = []
    for sent in doc.sents:
        for token in sent:
            text.append(token.text)
    tok_text = []
    for item in text:
        ap = item.find("'")
        if ap>=0:
            tok_text.append(item[0:ap])
            tok_text.append("'")
            tok_text.append(item[ap+1:len(item)])
        else:
            tok_text.append(item)
    text = []
    for item in tok_text:
        if len(item) > 0:
            text.append(item)

    text_ls = text[:]

    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0, [],[]
    else:

        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)
        # get importance score
        leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                    leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                      leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance score for word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set and len(text_ls[idx]) > 2:
                    words_perturb.append((idx, score))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))
        #return '', 0, orig_label, orig_label, 0, [], words_perturb
        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        #synonym_words, synonym_values, synonyms_dict = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, -1.0)
        # start replacing and attacking
        text_prime = text_ls[:]
        sims=[]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, score in words_perturb:
            #print(text_ls[idx])
            text_range_min, text_range_max = calc_window(idx, 3, 10, len_text)

            sliced_text = text_prime[text_range_min:text_range_max]
            #print(sliced_text)
            new_index = idx - text_range_min
            #print(sliced_text[new_index])
            masked_idx = new_index

            tokens, words, position = gen.convert_sentence_to_token(' '.join(sliced_text),1000,tokenizer)
            assert len(words)==len(position)

            len_tokens = len(tokens)

            mask_position = position[masked_idx]

            if isinstance(mask_position,list):
                feature = gen.convert_whole_word_to_feature(tokens, mask_position, 1000, tokenizer)
            else:
                feature = gen.convert_token_to_feature(tokens, mask_position, 1000, tokenizer)

            tokens_tensor = torch.tensor([feature.input_ids])
            token_type_ids = torch.tensor([feature.input_type_ids])
            attention_mask = torch.tensor([feature.input_mask])
            tokens_tensor = tokens_tensor.to('cuda')
            token_type_ids = token_type_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            #new_probs = predictor(new_texts, batch_size=batch_size)
            masked_lang_model.to('cuda')
            masked_lang_model.eval()
            ps = PorterStemmer()

            with torch.no_grad():
                prediction_scores  = masked_lang_model(tokens_tensor, token_type_ids,attention_mask)

            if isinstance(mask_position,list):
                predicted_top = prediction_scores[0, mask_position[0]].topk(50)
            else:
                predicted_top = prediction_scores[0, mask_position].topk(50)

            pre_tokens = tokenizer.convert_ids_to_tokens(predicted_top[1].cpu().numpy())
            synonyms_initial = gen.substitution_generation(words[masked_idx], pre_tokens, predicted_top[0].cpu().numpy(), ps, 50)
            new_texts = []
            avg = []
            synonyms = []
            assert words[masked_idx] == text_ls[idx]
            #print(synonyms)
            for candidate_word in synonyms_initial:
                if candidate_word in word_embedding and words[masked_idx] in word_embedding:
                    candidate_similarity = calc_similarity(word_embedding[words[masked_idx]],word_embedding[candidate_word])
                    avg.append(candidate_similarity)
                    #print(words[masked_idx], candidate_similarity, candidate_word)
                    if candidate_similarity >= 0.2:
                        new_texts.append(text_prime[:idx] + [candidate_word] + text_prime[min(idx + 1, len_text):])
                        synonyms.append(candidate_word)
                else:
                    new_texts.append(text_prime[:idx] + [candidate_word] + text_prime[min(idx + 1, len_text):])
                    synonyms.append(candidate_word)
            #print(len(new_texts))
            if len(new_texts) == 0:
                continue

            text_range_min, text_range_max = calc_window(idx, half_sim_score_window, sim_score_window, len_text)
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]
            sims.append(np.sum(semantic_sims)/len(semantic_sims))

            new_probs_mask = np.ones(len(new_texts)) #(orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask
            new_vals=semantic_sims*new_probs_mask
            index=[]
            mini=2
            for i in range(len(new_vals)):
                if new_vals[i] > 0:
                    index.append((new_vals[i],i))
            if len(index)==0:
                continue
            new_texts1=[new_texts[ind] for val,ind in index]
            #print(len(new_texts1))
            num_queries += len(new_texts1)
            if num_queries > qrs:
                return '', 0, orig_label, orig_label, 0, [],[]
            new_probs = predictor(new_texts1, batch_size=batch_size)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            pr=(orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            if np.sum(pr)>0 :
                text_prime[idx] = synonyms[index[pr.argmax()][1]] #synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label]
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[index[new_label_prob_argmin][1]]
                    num_changed += 1
            text_cache = text_prime[:]

            if fuzz.token_set_ratio(' '.join(text_ls) , ' '.join(text_cache)) < fuzz_val:
                return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries, words_perturb,sims
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries, words_perturb,sims


def random_attack(text_ls, true_label, predictor, perturb_ratio, stop_words_set, word2idx, idx2word, cos_sim,
                  sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15,
                  synonym_num=50, batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # randomly get perturbed words
        perturb_idxes = random.sample(range(len_text), int(len_text * perturb_ratio))
        words_perturb = [(idx, text_ls[idx]) for idx in perturb_idxes]

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]
            breakpoint()

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
            if fuzz.token_set_ratio(' '.join(text_ls),' '.join(text_cache)) < 80 :
                return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['wordLSTM', 'bert', 'wordCNN'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=5000,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")
    parser.add_argument("--target_dataset",
                        default="imdb",
                        type=str,
                        help="Dataset Name")
    parser.add_argument("--fuzz",
                        default=0,
                        type=int,
                        help="Word Pruning Value")
    parser.add_argument("--top_k_words",
                        default=1000000,
                        type=int,
                        help="Top K Words")
    parser.add_argument("--allowed_qrs",
                        default=1000000,
                        type=int,
                        help="Allowerd qrs")


    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # get data to attack
    texts, labels = dataloader.read_corpus(args.dataset_path,csvf=False)
    data = list(zip(texts, labels))
    data = data[:args.data_size] # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}
    sim_lis=[]
    word_embedding = defaultdict(list)

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            embedding = [float(num) for num in line.strip().split()[1:]]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1
                word_embedding[word] = embedding

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        with open(args.counter_fitting_cos_sim_path, "rb") as fp:
            sim_lis = pickle.load(fp)
        # load pre-computed cosine similarity matrix if provided
        #print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        #cos_sim = np.load(args.counter_fitting_cos_sim_path)
    else:
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        print(embeddings.T.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)

    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    avg=0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    wrds=[]
    s_queries=[]
    f_queries=[]
    sims_final=[]
    success=[]
    results=[]
    fails=[]
    scrs = []

    log_file = "results_context/"+args.target_model+"/"+args.target_dataset+"/log.txt"
    result_file = "results_context/"+args.target_model+"/"+args.target_dataset+"/results.csv"
    wts_file = "tfidf_weights/"+"tfidf-"+args.target_dataset+".csv"
    fail_file = "fails_tfidf/"+args.target_model+"/"+args.target_dataset+"/fails.csv"

    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')

    for idx, (text, true_label) in enumerate(data):
        #print(text)
        #print(true_label)
        if idx % 20 == 0:
            print(str(idx)+" Samples Done")
            print(len(success))
            print(np.mean(changed_rates))
        if args.perturb_ratio > 0.:
            new_text, num_changed, orig_label, \
            new_label, num_queries = random_attack(text, true_label, predictor,
                                                    args.perturb_ratio, stop_words_set,
                                                    word2idx, idx2word, cos_sim, sim_predictor=use,
                                                    sim_score_threshold=args.sim_score_threshold,
                                                    import_score_threshold=args.import_score_threshold,
                                                    sim_score_window=args.sim_score_window,
                                                    synonym_num=args.synonym_num,
                                                    batch_size=args.batch_size)
        else:
            new_text, num_changed, orig_label, \
            new_label, num_queries,pwrds,perts = attack(args.fuzz,args.top_k_words,args.allowed_qrs,
                                            wts_file,idx,text, true_label, predictor, stop_words_set,
                                            word2idx, idx2word, sim_lis, word_embedding , sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size)
        scrs.append(perts)
        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)

        if true_label != new_label:
            adv_failures += 1
            #f_queries.append(num_queries)

        changed_rate = 1.0 * num_changed / len(text)
        if true_label == orig_label and true_label != new_label:
            temp=[]
            s_queries.append(num_queries)
            success.append(idx)
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(text))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)
            wrds.append(pwrds)
            temp.append(idx)
            temp.append(' '.join(text))
            temp.append(new_text)
            temp.append(num_queries)
            temp.append(changed_rate * 100)
            results.append(temp)
            print("Attacked: "+str(idx))
        if true_label == orig_label and true_label == new_label:
            f_queries.append(num_queries)
            temp1=[]
            temp1.append(idx)
            temp1.append(' '.join(text))
            temp1.append(new_text)
            temp1.append(num_queries)
            fails.append(temp1)

    message = 'For target model {} on dataset window size {} with WP val {} top words {} qrs {} : ' \
              'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}\n'.format(args.target_model,
                                                                      args.sim_score_window,
                                                                      args.fuzz,
                                                                      args.top_k_words,args.allowed_qrs,
                                                                     (1-orig_failures/1000)*100,
                                                                     (1-adv_failures/1000)*100,
                                                                     np.mean(changed_rates)*100,
                                                                     np.mean(nums_queries))
    print(message)

    log=open(log_file,'a')
    log.write(message)
    with open(result_file,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)
    #with open('scores-mr.csv','w') as csvfile:
    #    csvwriter = csv.writer(csvfile)
    #    csvwriter.writerows(scrs)
#    with open(fail_file,'w') as csvfile:
#        csvwriter = csv.writer(csvfile)
#        csvwriter.writerows(fails)
    # writing the data rows
    print(avg)
    print(len(f_queries))
    print(f_queries)

    with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w') as ofile:
        for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
            ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

if __name__ == "__main__":
    main()

