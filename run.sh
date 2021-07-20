# train a model
python scripts/train.py --cfg_file=./exp/gas/vec/params.json

# train ppo policy
python scripts/run_dfa.py --cfg_file=./exp/gas/ppo/params.json