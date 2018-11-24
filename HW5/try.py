import tensorflow as tf
import numpy as np
import random
import os

from atari_emulator import AtariEmulator
from ale_python_interface import ALEInterface
from atari_emulator import AtariEmulator



BIN = "atari_roms/breakout.bin"
noops = 30
test_count = 1

def create_environment():
    ale_int = ALEInterface()
    ale_int.loadROM(str.encode(BIN))
    num_actions = len(ale_int.getMinimalActionSet())
    return AtariEmulator(BIN), num_actions


def choose_next_actions(num_actions, states, session):
    policy, value = session.run(['local_learning_2/actor_output_policy:0', 'local_learning_2/critic_output_out:0'], feed_dict = {'local_learning/input:0': states})
    policy = policy - np.finfo(np.float32).epsneg

    action_indices = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in policy]

    new_actions = np.eye(num_actions)[action_indices]

    return new_actions, value, policy


def run():
    environment, num_actions = create_environment()
    checkpoints_ = "pretrained/breakout/checkpoints/"
    with tf.Session() as sess:
        meta_ = os.path.join(checkpoints_, "haha.meta")

        saver = tf.train.import_meta_graph(meta_)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoints_))

        states = np.array([environment.get_initial_state()])
        if noops != 0:
            for _ in range(random.randint(0, noops)):
                state, _, _ = environment.next(environment.get_noop())

        episodes_over = np.zeros(test_count, dtype=np.bool)
        rewards = np.zeros(1, dtype=np.float32)
        while not all(episodes_over):
            actions, _, _ = choose_next_actions(num_actions, states, sess)

            state, reward, episode_over = environment.next(actions[0])
            states = np.array([state])
            rewards[0] += reward
            episodes_over[0] = episode_over

        print('Performed {} tests for breakout.'.format(test_count))
        print('Mean: {0:.2f}'.format(np.mean(rewards)))
        print('Min: {0:.2f}'.format(np.min(rewards)))
        print('Max: {0:.2f}'.format(np.max(rewards)))
        print('Std: {0:.2f}'.format(np.std(rewards)))

run()
