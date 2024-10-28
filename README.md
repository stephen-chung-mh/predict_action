
# Prediction Future Action of Reinforcement Learning Agents

This is a repo forked from the v1.21 of [Thinker repo](https://github.com/stephen-chung-mh/thinker/tree/v1.21). To reproduce the results in the paper, there are three main steps: 1. training the agents; 2. generating training data; 3. training the predictors. See the [installation](https://github.com/stephen-chung-mh/thinker/tree/v1.21#installation) section for details on installing the packages.

## 1. Training the Agents
First, switch to the `thinker` directory.

To train a default MuZero agent in Sokoban, run:
```bash
python train.py --xpid sokoban_muzero --mcts true --tree_carry false --rec_t 100 --actor_unroll_len 200 --auto_res False --env_n 256 --buffer_traj_len 20 --max_depth -1 --has_action_seq false --detect_dan_num 1 --total_steps 25000000
```
To train a default Thinker agent in Sokoban, run:
```bash
python train.py --xpid sokoban_thinker --detect_dan_num 1 --total_steps 25000000 
```
To train a default DRC agent in Sokoban, run:
```bash
python train.py --xpid sokoban_drc --drc true --actor_unroll_len 20 --reg_cost 0.01 --actor_learning_rate 4e-4 --entropy_cost 1e-2 --v_trace_lamb 0.97 --actor_adam_eps 1e-4 --detect_dan_num 1 --total_steps 25000000
```
To train a default IMPALA agent in Sokoban, run:
```bash
python train.py --xpid sokoban_impala --wrapper_type 1 --actor_unroll_len 20 --actor_learning_rate 3e-4 --see_real_state true --real_state_rnn false --tran_lstm_no_attn true --tran_layer_n 1 --detect_dan_num 1 --total_steps 25000000
```

## 2. Generating Training Data

By default, the trained agents will be stored in `logs/thinker/$XPID`. Now, switch to the `thinker/detect` directory and run the following to generate 50000 transitions (`$XPID` is the experiment id from above, such as `sokoban_muzero`) for the agent to be predicted:

```bash
python detect_gen.py --xpid $XPID --total_n 50000 --n 0 --env_n 64
```
You can remove the `--greedy` option to use a non-greedy policy, i.e., sampling actions from the policy instead of selecting the action with the largest probability.

## 3. Training the Predictors

The transitions should all be stored in `data/transition/$XPID-0` by default. To train a default predictor with only state-action input for event prediction, run:

```bash
python detect_train.py --txpid $TXPID --dxpid $XPID-0 --data_n 50000 --only_sa --no_rnn
```
For DRC and IMPALA, removing `--only_sa --no_rnn` will yield the simulation-based approach described in the paper. 
For DRC and IMPALA, removing `--only_sa --no_rnn` and adding `--see_hidden_state` will yield the inner-based approach described in the paper. 
For MuZero and Thinker, removing `--only_sa --no_rnn` will yield the inner-state approach described in the paper. 
Add the option `--pred_action` to switch to action prediction, and change the number in `--data_n` to change the training data size.

The predictor performance will be stored in the folder `data/detect_log/$TXPID`. Here, `$TXPID` is the predictor experiment ID and is arbitrary.

For other details of the code, please refer to the [original repo](https://github.com/stephen-chung-mh/thinker/tree/v1.21).

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Contact
For any questions or discussions, please contact me at mhc48@cam.ac.uk.