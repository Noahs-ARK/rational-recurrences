
LR=1
NUM_STEPS=300
TRAINER=sgd
REG_STR=0.01


python train_classifier.py --path /home/jessedd/data/amazon --embedding /home/jessedd/data/amazon/embedding --depth 1 --trainer ${TRAINER} --lr ${LR} --reg_strength ${REG_STR} --num_steps_debug ${NUM_STEPS} > ~/scratch/norms_${TRAINER}_steps=${NUM_STEPS}_lr=${LR}_regstr=${REG_STR}.txt


