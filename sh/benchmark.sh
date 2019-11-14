#!/bin/bash

python run.py --dataset mm1k  --struct mmmk --config rrinf --alpha 100 --geo 0.1 --lr 0.1  --epoch 50 --loop 1 --from 0 --simulate --seed 47
python run.py --dataset mmul  --struct mmul --config rrinf --alpha 100 --geo 0.1 --lr 0.01 --epoch 50 --loop 1 --from 0 --simulate --seed 47
python run.py --dataset mmmmr --struct mmmk --config rrinf --alpha 100 --geo 0.1 --lr 0.1  --epoch 50 --loop 1 --from 0 --simulate --seed 47
