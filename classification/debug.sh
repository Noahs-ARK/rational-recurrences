curr_dir="$PWD"
mkdir -p model
mkdir -p log

gpu_id=1
num_cpu=1
activation="tanh" # ['none', 'tanh', 'relu']
trainer="adam" # ['sgd', 'adam']
patience=30
lr=1e-3
lr_decay=0.0
batch_size=32
d=256
<<<<<<< HEAD
embed_dropout=0.0
rnn_dropout=0.0
dropout=0.0
depth=2
use_output_gate=False
fix_embed=True
bidirectional=False
use_layer_norm=False

semiring='max_plus'
=======
embed_dropout=0.4
rnn_dropout=0.3
dropout=0.3
depth=2
use_output_gate=False
fix_embed=True
use_layer_norm=False

semiring='plus_times'
>>>>>>> a1fd020... scripts

data=sst
path="data/sst"
embedding='data/sst/glove.300.sst.pruned'
<<<<<<< HEAD
model="rrnn" # ['rrnn', 'lstm']
gpu=False
=======
model="rrnn" # ['sopa', 'sru', 'lstm', 'qrnn']
gpu=True
>>>>>>> a1fd020... scripts
eval_ite=100
max_epoch=200
weight_decay=1e-6
clip_grad=5

export CUDA_VISIBLE_DEVICES=${gpu_id}
export OMP_NUM_THREADS=${num_cpu}

	#nohup \
<<<<<<< HEAD
	python3.5 train_classifier.py \
	--gpu=${gpu} \
        --use_layer_norm=${use_layer_norm} \
	--path=${path} \
	--embedding=${embedding} \
	--model=${model} \
        --semiring=${semiring} \
	--activation=${activation} \
	--batch_size=${batch_size} \
	--d=${d} \
        --embed_dropout=${embed_dropout} \
=======
	python3.6 train_classifier.py \
	--gpu=${gpu} \
    --use_layer_norm=${use_layer_norm} \
	--path=${path} \
	--embedding=${embedding} \
	--model=${model} \
    --semiring=${semiring} \
	--activation=${activation} \
	--batch_size=${batch_size} \
	--d=${d} \
    --embed_dropout=${embed_dropout} \
>>>>>>> a1fd020... scripts
	--dropout=${dropout} \
	--rnn_dropout=${rnn_dropout} \
	--depth=${depth} \
	--lr=${lr} \
	--lr_decay=${lr_decay} \
	--weight_decay=${weight_decay} \
	--clip_grad=${clip_grad} \
	--max_epoch=${max_epoch} \
<<<<<<< HEAD
        --use_output_gate=${use_output_gate} \
=======
    --use_output_gate=${use_output_gate} \
>>>>>>> a1fd020... scripts
	--fix_embedding=${fix_embed} \
	#> ${log_file}
