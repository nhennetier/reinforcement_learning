import gym
import numpy as np
import tensorflow as tf
import random

from utils import VFAConfig, ValueFunctionApproximation

class Config():
    #Optimization parameters
    minibatch_size = 1000
    epsilon = lambda self, x: 1/(1+x)

class Player():
    def __init__(self, config):
        self.config = config
        self.vfa_config = VFAConfig()
        self.vfa = ValueFunctionApproximation(self.vfa_config)

        self.nb_actions = 0
        self.nb_episodes = 0
        #Recorded sequences of the game for future optimizations of value functions.
        self.reset_replay_data()

    def collect_replay_data(self, prev_state, action, cur_state, reward):
        self.inputs.append(prev_state)
        self.actions.append([1 if action==i else 0 for i in self.vfa_config.action_space])
        self.outputs.append(cur_state)
        self.rewards.append(reward)

    def update_value_function(self, session):
        terminal_states = [int(x==1) for x in self.rewards]
        ids = np.random.choice(len(self.rewards), self.config.minibatch_size)
        self.vfa.run_batch(session,
            np.array(self.inputs)[ids],
            np.array(self.actions)[ids],
            np.array(self.outputs)[ids],
            np.array(terminal_states)[ids],
            np.array(self.rewards)[ids])

    def reset_replay_data(self):
        self.inputs = []
        self.actions = []
        self.outputs = []
        self.rewards = []

    def decide_action(self, session, inputs):
        self.nb_actions += 1

        opt_choice = self.vfa.best_next_actions(session,
            np.array([inputs]))[0]
        #epsilon-greedy exploration algorithm
        probs = [self.config.epsilon(self.nb_episodes/10) / len(self.vfa_config.action_space)
                 for _ in self.vfa_config.action_space]
        probs[opt_choice] += 1 - self.config.epsilon(self.nb_episodes/10)
        return list(np.random.multinomial(1, probs)).index(1)


if __name__=='__main__':
    env = gym.make('CartPole-v1')

    config = Config()
    player = Player(config)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        for i in range(2000):
            player.nb_episodes += 1
            new_observation = env.reset()
            for j in range(200):
                env.render()

                action = player.decide_action(session, new_observation)

                old_observation = new_observation
                new_observation, reward, done, _ = env.step(action)

                player.collect_replay_data(old_observation, action, new_observation, reward * (1-10*done))
                player.update_value_function(session)

                if done:
                    break

            if j==199:
                print('Victory !')
            else:
                print('Crash... Number of steps: %s' % j)

                    

