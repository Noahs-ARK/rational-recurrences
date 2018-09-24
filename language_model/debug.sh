curr_dir="$PWD"
mkdir -p model
mkdir -p log

<<<<<<< HEAD
gpu_id=2
=======
gpu_id=0
>>>>>>> a1fd020... scripts
num_cpu=1
activation="tanh" # ['none', 'tanh', 'relu']
patience=30
lr=1.0
lr_decay_epoch=150
lr_decay=0.98
batch_size=32
<<<<<<< HEAD
d=256
input_dropout=0.0
output_dropout=0.0
rnn_dropout=0.0
dropout=0.0
depth=2
use_output_gate=False

model="rrnn" # ['rrnn', 'lstm']
semiring="plus_times"
gpu=False
=======
d=910
input_dropout=0.65
output_dropout=0.65
rnn_dropout=0.2
dropout=0.2
depth=3
use_output_gate=False

model="rrnn" # ['sopa', 'sru', 'lstm']
semiring="plus_times"
gpu=True
>>>>>>> a1fd020... scripts
eval_ite=100
unroll_size=35
weight_decay=1e-5
clip_grad=5

# sru 2 layers, 10m: 720
# bigram 2 layers, 10m: 610
# unigram 2 layers, 10m: 650
export CUDA_VISIBLE_DEVICES=${gpu_id}
export OMP_NUM_THREADS=${num_cpu}

<<<<<<< HEAD
    python3.5 train_lm.py \
    --gpu=${gpu} \
	--train=data/dev \
	--dev=data/dev \
	--test=data/dev \
=======
    python3.6 train_lm.py \
    --gpu=${gpu} \
	--train=data/train \
	--dev=data/dev \
	--test=data/test \
>>>>>>> a1fd020... scripts
	--model=${model} \
    --semiring=${semiring} \
	--activation=${activation} \
    --use_output_gate=${use_output_gate} \
	--batch_size=${batch_size} \
	--eval_ite=${eval_ite} \
	--unroll_size=${unroll_size} \
	--d=${d} \
	--input_dropout=${input_dropout} \
	--output_dropout=${output_dropout} \
	--rnn_dropout=${rnn_dropout} \
	--dropout=${dropout} \
	--depth=${depth} \
	--lr=${lr} \
	--lr_decay=${lr_decay} \
	--lr_decay_epoch=${lr_decay_epoch} \
	--weight_decay=${weight_decay} \
	--clip_grad=${clip_grad} \
    --patience=${patience} \
