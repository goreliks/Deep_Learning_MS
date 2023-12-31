#!/bin/bash

python -m hw2.experiments run-exp -n test -K 32 -L 8 -P 8 -H 100 -M resnet --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4_L8_K32
python -m hw2.experiments run-exp -n test -K 32 -L 16 -P 8 -H 100 -M resnet --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4_L16_K32
python -m hw2.experiments run-exp -n test -K 32 -L 32 -P 8 -H 100 -M resnet --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4_L32_K32
python -m hw2.experiments run-exp -n test -K 64 128 256 -L 2 -P 4 -H 100 -M resnet  --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4_L2_K64-128-256
python -m hw2.experiments run-exp -n test -K 64 128 256 -L 4 -P 4 -H 100 -M resnet  --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4_L4_K64-128-256
python -m hw2.experiments run-exp -n test -K 64 128 256 -L 8 -P 4 -H 100 -M resnet  --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4_L8_K64-128-256