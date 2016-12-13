import random
import numpy as np
import tensorflow as tf

class Config():
    '''Configuration of the game.'''
    nb_decks = 6
    #Fraction of deck to reach before shuffling
    shuffling = 0.7
    #Initial stake of each player
    current_stake = 1000000
    #Possible actions: 0=stand, 1=hit
    action_space = np.arange(2)
    #Optimization parameters
    batch_size = 100
    lr = lambda self, x: 1/(1+x)
    epsilon = lambda self, x: 1/(1+x)
    discount = 1/(1+1e-1)
    #Whether to print actions in the game
    verbose = False

class CardsConfig():
    def __init__(self):
        self.faces = ['Ace', 'King', 'Queen', 'Jack']
        self.units = [str(x) for x in range(2, 11)]

    def decode(self, card, total=0):
        '''Returns the value of considered card.'''
        if card[:3] == 'Ace':
            if total > 10:
                return 1
            else:
                return 11
        elif card[:4] in self.faces or card[:5] in self.faces:
            return 10
        else:
            return int(card[:2])

    def sum_cards(self, cards):
        '''Returns the total value of a hand.'''
        total = 0
        aces = [card for card in cards if card[:3]=='Ace']
        cards = [card for card in cards if card[:3]!='Ace']
        for card in cards:
            total += self.decode(card, total)
        #Aces are added at the end to prevent bust (value of 11 or 1).
        for ace in aces:
            total += self.decode(ace, total)
        return total

    def display_cards(self, cards):
        result = cards[0]
        for card in cards[1:]:
            result += (' + %s' % card)
        result += (' = %s' % self.sum_cards(cards))
        return result

class Deck():
    def __init__(self, config, cards_config):
        self.config = config
        self.cards_config = cards_config
        self.new_deck()

    def new_deck(self):
        '''Returns a new shuffled deck.'''
        colors = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
        self.cards = self.config.nb_decks * ([u+' of '+c for u in self.cards_config.units for c in colors] \
                                        + [u+' of '+c for u in self.cards_config.faces for c in colors])
        random.shuffle(self.cards)
        
    def draw(self):
        drawn_card = self.cards[0]
        self.cards = self.cards[1:]
        return drawn_card


class BlackJackGame(Deck):
    def __init__(self, session, config, cards_config, player):
        self.config = config
        self.cards_config = cards_config
        self.player = player
        self.deck = Deck(self.config, self.cards_config)

        while self.player.current_stake > 0:
            #New deck and new counts if threshold is reached
            if len(self.deck.cards) < 52 * self.config.nb_decks * (1 - self.config.shuffling):
                self.deck.new_deck()
                self.player.nb_decks += 1
                self.player.nb_cards = 0
                self.player.current_hilo = 0
                
                if self.config.verbose:
                    print('Automatic shuffling of cards.\n')

            #New play
            self.play(session)

            #Optimization step of value functions
            if self.player.nb_plays % self.config.batch_size == 0:
                print('Nb of plays: %s' % self.player.nb_plays)
                terminal_states = [1 if (self.player.outputs[i][0]<=21 \
                                         and (self.player.actions[i]%len(self.config.action_space))==1)
                                   else 0 for i in range(len(self.player.outputs))]
                self.player.update_value_function(session,
                                                  np.array(self.player.inputs),
                                                  np.array(self.player.actions),
                                                  np.array(self.player.outputs),
                                                  np.array(terminal_states),
                                                  np.array(self.player.rewards))
                self.player.inputs = []
                self.player.actions = []
                self.player.outputs = []
                self.player.rewards = []
                
            #End the game after a given number of plays.
            if self.player.nb_plays > 100000:
                break

        if self.player.current_stake <= 0:
            print('You lost all your initial stake, end of the game.')
        else:
            print('End of the game.')

    def display_player(self):
        if self.config.verbose:
            print('Player: %s ' % (self.cards_config.display_cards(self.player.cards)))

    def display_dealer(self):
        if self.config.verbose:
            print('Dealer: %s ' % (self.cards_config.display_cards(self.dealer)))
        
    def play(self, session):
        #Betting round
        bet = self.player.decide_bet()
        self.player.current_stake -= bet

        #New cards for player and dealer
        self.player.ace = False
        self.player.cards = [self.deck.draw()]
        self.dealer = [self.deck.draw()]
        self.player.cards.append(self.deck.draw())

        #Counts updates
        for card in self.player.cards:
            self.player.update_counts(card, 1)
        self.player.update_counts(self.dealer[0], 0)

        self.display_dealer()

        #Loop until terminal state
        while True:
            self.display_player()

            #Keep track of previous state (hilo count and ace in the hand)
            if self.player.cards[-1][:3] == 'Ace':
                prev_hilo = self.player.current_hilo + 1
                prev_ace = np.any([card[:3]=='Ace' for card in self.player.cards[:len(self.player.cards)-1]])
            elif self.cards_config.decode(self.player.cards[-1]) >= 10:
                prev_hilo = self.player.current_hilo + 1
                prev_ace = self.player.ace
            elif self.cards_config.decode(self.player.cards[-1]) < 7:
                prev_hilo = self.player.current_hilo - 1
                prev_ace = self.player.ace
            else:
                prev_hilo = self.player.current_hilo
                prev_ace = self.player.ace

            #Check whether player is in a terminal state (Blackjack or bust)
            total_player = self.cards_config.sum_cards(self.player.cards)
            if len(self.player.cards) == 2 and total_player == 21:
                victory = 1
                if self.config.verbose:
                    print('BlackJack ! You win.\n')
                break
            elif total_player > 21:
                victory = 0
                #Collect data for future optimizations
                self.player.collect_replay_data(
                    self.cards_config.sum_cards(self.player.cards[:len(self.player.cards)-1]),
                    self.cards_config.sum_cards(self.dealer),
                    prev_ace,
                    1,
                    self.cards_config.sum_cards(self.player.cards),
                    self.cards_config.sum_cards(self.dealer),
                    self.player.ace,
                    -1)

                if self.config.verbose:
                    print('You went bust. You loose.\n')
                break
            
            #Collect data if player has more than 2 cards and is not in a terminal state
            #(player has drawn a card in previous state)
            if len(self.player.cards) > 2:
                self.player.collect_replay_data(
                    self.cards_config.sum_cards(self.player.cards[:len(self.player.cards)-1]),
                    self.cards_config.sum_cards(self.dealer),
                    prev_ace,
                    1,
                    self.cards_config.sum_cards(self.player.cards),
                    self.cards_config.sum_cards(self.dealer),
                    self.player.ace,
                    0)

            #Player decides to draw or not given his current state
            action = self.player.decide_draw(session,
                                             self.cards_config.sum_cards(self.player.cards),
                                             self.cards_config.sum_cards(self.dealer),
                                             self.player.ace)

            if action:
                self.player.cards.append(self.deck.draw())
                self.player.update_counts(self.player.cards[-1], 1)
            else:
                #Dealer draws cards until terminal state
                while self.cards_config.sum_cards(self.dealer) < 17:
                    self.dealer.append(self.deck.draw())
                    self.player.update_counts(self.dealer[-1], 0)
                    self.display_dealer()
                    
                if self.cards_config.sum_cards(self.dealer) > 21:
                    if self.config.verbose:
                        print('The dealer went bust. You win !\n')
                    victory = 1
                elif self.cards_config.sum_cards(self.dealer) >= total_player:
                    if self.config.verbose:
                        print('The dealer is %s and you\'re %s. You loose.\n' \
                            % (self.cards_config.sum_cards(self.dealer), total_player))
                    victory = 0
                else:
                    if self.config.verbose:
                        print('The dealer is %s and you\'re %s. You win !\n' \
                            % (self.cards_config.sum_cards(self.dealer), total_player))
                    victory = 1

                #Collect data of terminal state
                self.player.collect_replay_data(
                    self.cards_config.sum_cards(self.player.cards),
                    self.cards_config.sum_cards(self.dealer[:1]),
                    self.player.ace,
                    0,
                    self.cards_config.sum_cards(self.player.cards),
                    self.cards_config.sum_cards(self.dealer[:1]),
                    self.player.ace,
                    2*victory-1)
                break

        #Update player's current stake
        self.player.current_stake += 2*bet*victory

        self.player.stakes.append(self.player.current_stake)
        self.player.nb_plays += 1
        if self.config.verbose:
            print('\nCurrent stake: %s\n' % self.player.current_stake)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    from players import Baseline_Player, BasicStrategy_Player, SARSA_RLPlayer, Linear_RLPlayer, Deep_RLPlayer

    config = Config()
    cards_config = CardsConfig()

    with tf.Session() as session:
        bl_player = Baseline_Player(session, config, cards_config)
        BlackJackGame(session, config, cards_config, bl_player)

        bs_player = BasicStrategy_Player(session, config, cards_config)
        BlackJackGame(session, config, cards_config, bs_player)

        sarsa_rlplayer = SARSA_RLPlayer(session, config, cards_config)
        for i in range(10):
            sarsa_rlplayer.reset_stake()
            BlackJackGame(session, config, cards_config, sarsa_rlplayer)

        linear_rlplayer = Linear_RLPlayer(session, config, cards_config)
        for i in range(10):
            linear_rlplayer.reset_stake()
            BlackJackGame(session, config, cards_config, linear_rlplayer)
        linear_rlplayer.vfa.vfa_config.lr = 1e-3
        for i in range(10):
            linear_rlplayer.reset_stake()
            BlackJackGame(session, config, cards_config, linear_rlplayer)

        deep_rlplayer = Deep_RLPlayer(session, config, cards_config)
        for i in range(20):
            deep_rlplayer.reset_stake()
            BlackJackGame(session, config, cards_config, deep_rlplayer)
        deep_rlplayer.vfa.vfa_config.lr = 1e-3
        for i in range(20):
            deep_rlplayer.reset_stake()
            BlackJackGame(session, config, cards_config, deep_rlplayer)

    plt.plot(np.arange(len(deep_rlplayer.stakes)), deep_rlplayer.stakes, label='Deep Q-Network')
    plt.plot(np.arange(len(linear_rlplayer.stakes)), linear_rlplayer.stakes, label='Linear Q-Network')
    plt.plot(np.arange(len(sarsa_rlplayer.stakes)), sarsa_rlplayer.stakes, label='SARSA')
    plt.plot(np.arange(len(bl_player.stakes)), bl_player.stakes, label='Baseline')
    plt.plot(np.arange(len(bs_player.stakes)), bs_player.stakes, label='Basic Strategy')
    plt.legend()
    plt.show()