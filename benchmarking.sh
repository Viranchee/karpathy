python3 makemore/makemore.py \
  --input-file names.txt \
  --work-dir work \
  --seed 42 \
  --type transformer \
  --n-head 8 \
  --n-layer 4 \
  --n-embd 1024 \
  --n-embd2 1024 \
  --device mps \
  --batch-size 512 \
  --resume \
  
  # --max-steps 1000 \
  # --learning-rate 0.005 \

# {'input_file': 'names.txt'
# 'work_dir': 'work'
# 'resume': False
# 'sample_only': False
# 'num_workers': 4
# 'max_steps': 100000
# 'device': 'mps'
# 'seed': 42
# 'top_k': -1
# 'type': 'transformer'
# 'n_layer': 4
# 'n_head': 4
# 'n_embd': 64
# 'n_embd2': 64
# 'batch_size': 32
# 'learning_rate': 0.0005
# 'weight_decay': 0.01}
# number of examples in the dataset: 1356048
# max word length: 15
# number of unique characters in the vocabulary: 26
# vocabulary:
# abcdefghijklmnopqrstuvwxyz
# split up the dataset into 1355048 training examples and 1000 test examples
# dataset determined that: vocab_size=27, block_size=16
# number of parameters: 0.20M
# model #params: 204544