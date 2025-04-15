python3 scripts/train.py --args.load conf/ablations/baseline.yml --save_path runs/0410

exp1: python3 scripts/train_dac.py --args.load conf/exps/dac_512_9.yml --save_path exps/DAC_512_9
exp2: python3 scripts/train_dac.py --args.load conf/exps/dac_512_2.yml --save_path exps/DAC_512_2

exp3: python3 scripts/train.py --args.load conf/exps/pyramid_p222_n555.yml --save_path exps/pyramid_p222_n555


python3 scripts/get_samples.py --args.load conf/exps/pyramid_p222_n555.yml
