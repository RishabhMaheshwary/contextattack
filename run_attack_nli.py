import os

command = 'python3 attack_nli.py --dataset_path data/snli ' \
          '--target_model bert ' \
          '--target_model_path BERT/results/snli ' \
          '--counter_fitting_cos_sim_path mat.txt ' \
          '--word_embeddings_path /scratch/glove.840B.300d.txt '\
          '--counter_fitting_embeddings_path counter-fitted-vectors.txt ' \
          '--USE_cache_path "nli_cache" ' \

os.system(command)
command = 'python3 attack_nli.py --dataset_path data/mnli ' \
          '--target_model bert ' \
          '--target_model_path BERT/results/mnli ' \
          '--counter_fitting_cos_sim_path mat.txt ' \
          '--word_embeddings_path /scratch/glove.840B.300d.txt '\
          '--counter_fitting_embeddings_path counter-fitted-vectors.txt ' \
          '--USE_cache_path "nli_cache" ' \

os.system(command)
