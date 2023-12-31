#!/bin/bash

python -m hw2.experiments run-exp -n test -K 32 -L 2 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L2_K32
python -m hw2.experiments run-exp -n test -K 64 -L 2 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L2_K64
python -m hw2.experiments run-exp -n test -K 128 -L 2 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L2_K128
python -m hw2.experiments run-exp -n test -K 256 -L 2 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L2_K256
python -m hw2.experiments run-exp -n test -K 32 -L 4 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L4_K32
python -m hw2.experiments run-exp -n test -K 64 -L 4 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L4_K64
python -m hw2.experiments run-exp -n test -K 128 -L 4 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L4_K128
python -m hw2.experiments run-exp -n test -K 256 -L 4 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L4_K256
python -m hw2.experiments run-exp -n test -K 32 -L 8 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L8_K32
python -m hw2.experiments run-exp -n test -K 64 -L 8 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L8_K64
python -m hw2.experiments run-exp -n test -K 128 -L 8 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L8_K128
python -m hw2.experiments run-exp -n test -K 256 -L 8 -P 4 -H 100 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L8_K256