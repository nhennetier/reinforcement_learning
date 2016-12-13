import numpy as np
import tensorflow as tf
import random

from utils import VFAConfig, LinearVFA, NeuralNetworkVFA

class BlackJackPlayer():
    def __init__(self, session, config, cards_config):
        self.config = config
        self.cards_config = cards_config
        self.current_stake = self.config.current_stake
        self.current_hilo = 0
        self.nb_cards = 0
        self.nb_plays = 0
        self.nb_decks = 0
        self.stakes = []

        #Recorded sequences of the game for future optimizations of value functions.
        self.player.inputs = []
        self.player.actions = []
        self.player.outputs = []
        self.player.rewards = []

    def reset_stake(self):
        self.current_stake = self.config.current_stake
        self.nb_plays = 0
        self.stakes = []

    def update_counts(self, card, player):
        self.nb_cards += 1
        if self.cards_config.decode(card) < 7:
            self.current_hilo += 1
        elif card[:3] == 'Ace':
            self.current_hilo -= 1
            if player:
                self.ace = 1
        elif self.cards_config.decode(card) >= 10:
            self.current_hilo -= 1

    def collect_replay_data(self, prev_player, prev_dealer, prev_ace, action, player, dealer, ace, reward):
        pass

    def update_value_function(self, session, inputs, actions, outputs, terminal_states, rewards):
        pass

    def decide_draw(self, session, player, dealer, ace):
        raise NotImplementedError

    def decide_bet(self):
        raise NotImplementedError

    def print_policy(self, session):
        policy = np.zeros((18, 10))
        for player in range(4, 22):
            for dealer in range(2, 12):
                policy[21 - player, dealer - 2] = self.decide_draw(session, player, dealer, 0)
        print(policy)

class Baseline_Player(BlackJackPlayer):
    '''Imitating the dealer's strategy.'''
    def decide_draw(self, session, player, dealer, ace):
        return int(self.cards_config.sum_cards(self.cards) < 17)

    def decide_bet(self):
        return 1

class BasicStrategy_Player(BlackJackPlayer):
    '''Basic strategy player.
    Know optimal policy using player's and dealer's count as inputs.'''
    def decide_draw(self, session, player, dealer, ace):
        if player > 16:
            return 0
        elif player < 12:
            return 1
        elif dealer > 6:
            return 1
        elif player == 12 and dealer < 4:
            return 1
        else:
            return 0

    def decide_bet(self):
        return 1

class SARSA_RLPlayer(BlackJackPlayer):
    '''RL Player using TD(0) SARSA algorithm with a table-lookup value function.'''
    def __init__(self, session, config, cards_config):
        BlackJackPlayer.__init__(self, session, config, cards_config)
        self.value_function = np.zeros((28, 10, 2, len(self.config.action_space)))

    def collect_replay_data(self, prev_player, prev_dealer, prev_ace, action, player, dealer, ace, reward):
        self.inputs.append([prev_player, prev_dealer, prev_ace])
        self.actions.append(action)
        self.outputs.append([player, dealer, ace])
        self.rewards.append(reward)

    def update_value_function(self, session, inputs, actions, outputs, terminal_states, rewards):
        for i in range(inputs.shape[0]):
            current_player, current_dealer, current_ace = inputs[i]
            next_player, next_dealer, next_ace = outputs[i]

            current_action = actions[i]
            reward = rewards[i]
            if current_action == 0:
                next_action = 0
            else:
                next_action = self.decide_draw(session, next_player, next_dealer, next_ace)

            if current_action == 1:
                Q_new = self.value_function[31 - next_player,
                                            next_dealer - 2,
                                            next_ace,
                                            next_action]
            else:
                Q_new = 0

            Q_old = self.value_function[31 - current_player,
                                        current_dealer - 2,
                                        current_ace,
                                        current_action]

            self.value_function[31 - current_player,
                                current_dealer - 2,
                                current_ace,
                                current_action] += self.config.lr(int(self.nb_decks/500)) \
                                    * (reward + self.config.discount * Q_new - Q_old)

    def decide_draw(self, session, player, dealer, ace):
        if player > 21:
            return 0
        else:
            values = self.value_function[31 - player,
                                         dealer - 2,
                                         ace,
                                         :]
            opt_choice = np.argmax(values)
            #epsilon-greedy exploration algorithm
            probs = [self.config.epsilon(int(self.nb_decks/500)) / len(self.config.action_space)
                     for _ in self.config.action_space]
            probs[opt_choice] += 1 - self.config.epsilon(int(self.nb_decks/500))
            return list(np.random.multinomial(1, probs)).index(1)

    def decide_bet(self):
        return 1


class Linear_RLPlayer(BlackJackPlayer):
    '''RL Player using TD(0) SARSA algorithm with a linear value function approximation.'''
    def __init__(self, session, config, cards_config):
        BlackJackPlayer.__init__(self, session, config, cards_config)
        self.vfa_config = VFAConfig()
        self.vfa = LinearVFA(self.vfa_config)
        session.run(tf.initialize_all_variables())

    def collect_replay_data(self, prev_player, prev_dealer, prev_ace, action, player, dealer, ace, reward):
        inputs = [1 if ((i%10 + 4) == prev_player
                    and (int(i/10)+2) == prev_dealer
                    and int(i/280) == prev_ace)
                  else 0
                  for i in range(560)]
        self.inputs.append(inputs)
        self.actions.append(len(self.config.action_space)*len(self.actions) + action)
        outputs = [1 if ((i%10 + 4) == player
                    and (int(i/10)+2) == dealer
                    and int(i/280) == ace)
                   else 0
                   for i in range(560)]
        self.outputs.append(outputs)
        self.rewards.append(reward)

    def update_value_function(self, session, inputs, actions, outputs, terminal_states, rewards):
        self.vfa.run_batch(session, inputs, actions, outputs, terminal_states, rewards)
        
    def decide_draw(self, session, player, dealer, ace):
        if player > 21:
            return 0
        else:
            inputs = [1 if ((i%10 + 4) == player
                        and (int(i/10)+2) == dealer
                        and int(i/280) == ace)
                      else 0
                      for i in range(560)]
            opt_choice = self.vfa.best_next_actions(session,
                np.array([inputs]))[0]
            #epsilon-greedy exploration algorithm
            probs = [self.config.epsilon(int(self.nb_decks/100)) / len(self.config.action_space)
                     for _ in self.config.action_space]
            probs[opt_choice] += 1 - self.config.epsilon(int(self.nb_decks/100))
            return list(np.random.multinomial(1, probs)).index(1)

    def decide_bet(self):
        return 1


class Deep_RLPlayer(BlackJackPlayer):
    '''RL Player using TD(0) SARSA algorithm with a Deep Q-Network as value function approximation.'''
    def __init__(self, session, config, cards_config):
        BlackJackPlayer.__init__(self, session, config, cards_config)
        self.vfa_config = VFAConfig()
        self.vfa = NeuralNetworkVFA(self.vfa_config)
        session.run(tf.initialize_all_variables())

    def collect_replay_data(self, prev_player, prev_dealer, prev_ace, action, player, dealer, ace, reward):
        inputs = [1 if ((i%10 + 4) == prev_player
                    and (int(i/10)+2) == prev_dealer
                    and int(i/280) == prev_ace)
                  else 0
                  for i in range(560)]
        self.inputs.append(inputs)
        self.actions.append(len(self.config.action_space)*len(self.actions) + action)
        outputs = [1 if ((i%10 + 4) == player
                    and (int(i/10)+2) == dealer
                    and int(i/280) == ace)
                   else 0
                   for i in range(560)]
        self.outputs.append(outputs)
        self.rewards.append(reward)

    def update_value_function(self, session, inputs, actions, outputs, terminal_states, rewards):
        self.vfa.run_batch(session, inputs, actions, outputs, terminal_states, rewards)
        
    def decide_draw(self, session, player, dealer, ace):
        if player > 21:
            return 0
        else:
            inputs = [1 if ((i%10 + 4) == player
                        and (int(i/10)+2) == dealer
                        and int(i/280) == ace)
                      else 0
                      for i in range(560)]
            opt_choice = self.vfa.best_next_actions(session,
                np.array([inputs]))[0]
            #epsilon-greedy exploration algorithm
            probs = [self.config.epsilon(int(self.nb_decks/100)) / len(self.config.action_space)
                     for _ in self.config.action_space]
            probs[opt_choice] += 1 - self.config.epsilon(int(self.nb_decks/100))
            return list(np.random.multinomial(1, probs)).index(1)

    def decide_bet(self):
        return 1
