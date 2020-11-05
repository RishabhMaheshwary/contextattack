import os

command = 'python3 transfer.py --dataset_path data/imdb ' \
          '--target_model bert ' \
          '--target_model_path BERT/results/imdb ' \
          '--word_embeddings_path glove.6B.200d.txt ' \
          '--counter_fitting_cos_sim_path mat.txt ' \
          '--counter_fitting_embeddings_path counter-fitted-vectors.txt ' \
          '--USE_cache_path " " '

os.system(command)

