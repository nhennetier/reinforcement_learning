import gym
import numpy as np
import tensorflow as tf
import random

from utils import VFAConfig, ValueFunctionApproximation

class Config():
    nb_steps = 10
    #Optimization parameters
    minibatch_size = 32
    update_frequency = 300
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
        self.reset_temp_data()

    def reset_replay_data(self):
        self.inputs = []
        self.actions = []
        self.outputs = []
        self.rewards = []

    def reset_temp_data(self):
        self.temp_inputs = []
        self.temp_actions = []
        self.temp_rewards = []

    def resize_temp_data(self):
        self.temp_inputs = self.temp_inputs[1:]
        self.temp_actions = self.temp_actions[1:]
        self.temp_rewards = self.temp_rewards[1:]

    def collect_temp_data(self, state, action, reward):
        self.temp_inputs.append(state)
        self.temp_actions.append([1 if action==i else 0 for i in self.vfa_config.action_space])
        self.temp_rewards.append(reward)

    def collect_replay_data(self):
        self.inputs.append(self.temp_inputs[0])
        self.actions.append(self.temp_actions[0])
        self.outputs.append(self.temp_inputs[-1])
        self.rewards.append(sum([x * self.vfa_config.discount**i
                                 for i,x in enumerate(self.temp_rewards)]))
        self.resize_temp_data()

    def update_value_function(self, session):
        if len(self.rewards) > 0:
            terminal_states = [int(x<0) for x in self.rewards]
            ids = np.random.choice(len(self.rewards), self.config.minibatch_size)
            self.vfa.run_batch(session,
                np.array(self.inputs)[ids],
                np.array(self.actions)[ids],
                np.array(self.outputs)[ids],
                np.array(terminal_states)[ids],
                np.array(self.rewards)[ids])

    def update_target_network(self, session):
        self.vfa.update_target_weights(session)

    def decide_action(self, session, inputs):
        self.nb_actions += 1

        opt_choice = self.vfa.best_next_actions(session,
            np.array([inputs]))[0]
        #epsilon-greedy exploration algorithm
        probs = [self.config.epsilon(self.nb_episodes) / len(self.vfa_config.action_space)
                 for _ in self.vfa_config.action_space]
        probs[opt_choice] += 1 - self.config.epsilon(self.nb_episodes)
        return list(np.random.multinomial(1, probs)).index(1)


if __name__=='__main__':
    env = gym.make('MountainCar-v0')

    config = Config()
    player = Player(config)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        for i in range(500):
            new_observation = env.reset()
            t = 0
            while True:
                t += 1
                env.render()

                action = player.decide_action(session, new_observation)

                old_observation = new_observation
                new_observation, reward, done, _ = env.step(action)

                player.collect_temp_data(old_observation, action, reward)

                if len(player.temp_rewards) >= player.config.nb_steps:
                    player.collect_replay_data()

                player.update_value_function(session)
                
                if player.nb_actions % player.config.update_frequency == 0:
                    player.update_target_network(session)

                if done:
                    while len(player.temp_rewards) > 0:
                        player.collect_replay_data()
                        player.update_value_function(session)

                    print('That\'s a victory ! Number of steps: %s' % t)
                    player.nb_episodes += 1
                    break

