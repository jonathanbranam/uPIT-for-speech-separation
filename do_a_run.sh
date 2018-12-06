./compute_cmvn.py conf/2spk/rel100/rel100_tr_mix.scp conf/2spk/rel100/cmvn.dict --wav-base-dir data
./run_pit.py --num-epoches 2 --config conf/rel4.yaml --wav-base-dir data --checkpoint tune
./separate.py --dump-dir rel100 conf/rel100.yaml ~/Google\ Drive/wsj0-mix-subset/tune/2spk_pit_a_rel100/epoch.50.pkl conf/2spk/rel100/rel100_tt_mix.scp --wav-base-dir data
