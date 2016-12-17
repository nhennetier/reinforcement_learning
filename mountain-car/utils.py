import tensorflow as tf

class VFAConfig(object):
    hidden_size = 4
    num_hidden_layers = 1
    state_dim = 2
    action_space = [0, 1, 2]
    lr = 1e-3
    discount = 1/(1+1e-1)
    alpha_reg = 1e-4
    beta_reg = 1e-4
    nb_steps = 10

class ValueFunctionApproximation():
    '''Base class for value Function Approximation. Shouldn't be used separately.'''
    def __init__(self, vfa_config):
        self.vfa_config = vfa_config
        
        self.add_placeholders()
        self.add_variables()
        self.Q_inputs, _, _ = self.add_comp_graph(self.input_placeholder,
                                                  self.input_weights,
                                                  self.input_biases)
        self.Q_targets, self.reg_l1, self.reg_l2 = self.add_comp_graph(self.output_placeholder,
                                                                       self.target_weights,
                                                                       self.target_biases)
        self.next_actions = tf.argmax(self.Q_inputs, 1)
        self.Q_inputs = tf.reduce_sum(tf.mul(self.Q_inputs, self.action_placeholder), 1)
        self.max_Q_targets = tf.reduce_max(self.Q_targets, 1)
        self.max_Q_targets = tf.mul(self.max_Q_targets, self.terminal_placeholder)
        
        self.calculate_loss = self.add_loss_op()
        self.train_step = self.add_training_op(self.calculate_loss)

        self.updates = []
        for i in range(self.vfa_config.num_hidden_layers + 2):
            self.updates.append(tf.assign(self.target_weights[i], self.input_weights[i]))
            self.updates.append(tf.assign(self.target_biases[i], self.input_biases[i]))

    def add_placeholders(self):
        #Current state
        self.input_placeholder = tf.placeholder(tf.float32,
            shape=[None, self.vfa_config.state_dim],
            name='Input')
        #Current action
        self.action_placeholder = tf.placeholder(tf.float32,
            shape=[None, len(self.vfa_config.action_space)],
            name='Action')
        #Next state
        self.output_placeholder = tf.placeholder(tf.float32,
            shape=[None, self.vfa_config.state_dim],
            name='Output')
        #Dummy variables, whether the next state is terminal
        self.terminal_placeholder = tf.placeholder(tf.float32,
            shape=[None],
            name='Terminal')
        #Next reward
        self.reward_placeholder = tf.placeholder(tf.float32,
            shape=[None],
            name='Reward')

    def add_variables(self):
        self.input_weights = [tf.Variable(tf.random_normal([self.vfa_config.state_dim,
                                                            self.vfa_config.hidden_size]),
                                          dtype=tf.float32)]
        self.input_weights += [tf.Variable(tf.random_normal([self.vfa_config.hidden_size,
                                                             self.vfa_config.hidden_size]),
                                           dtype=tf.float32)
                               for _ in range(self.vfa_config.num_hidden_layers)]
        self.input_weights += [tf.Variable(tf.random_normal([self.vfa_config.hidden_size,
                                                             len(self.vfa_config.action_space)]),
                                           dtype=tf.float32)]
        self.input_biases = [tf.Variable(tf.random_normal([self.vfa_config.hidden_size]),
                                         dtype=tf.float32)
                             for _ in range(self.vfa_config.num_hidden_layers + 1)]
        self.input_biases += [tf.Variable(tf.random_normal([len(self.vfa_config.action_space)]),
                                          dtype=tf.float32)]
        self.target_weights = [tf.Variable(tf.random_normal([self.vfa_config.state_dim,
                                                             self.vfa_config.hidden_size]),
                                           dtype=tf.float32)]
        self.target_weights += [tf.Variable(tf.random_normal([self.vfa_config.hidden_size,
                                                              self.vfa_config.hidden_size]),
                                            dtype=tf.float32)
                               for _ in range(self.vfa_config.num_hidden_layers)]
        self.target_weights += [tf.Variable(tf.random_normal([self.vfa_config.hidden_size,
                                                              len(self.vfa_config.action_space)]),
                                            dtype=tf.float32)]
        self.target_biases = [tf.Variable(tf.random_normal([self.vfa_config.hidden_size]),
                                          dtype=tf.float32)
                             for _ in range(self.vfa_config.num_hidden_layers + 1)]
        self.target_biases += [tf.Variable(tf.random_normal([len(self.vfa_config.action_space)]),
                                           dtype=tf.float32)]

    def add_comp_graph(self, inputs, weights, biases):
        reg_l1 = 0
        reg_l2 = 0
        hidden_state = inputs
        
        for step in range(self.vfa_config.num_hidden_layers + 2):
            hidden_state = tf.nn.tanh(tf.matmul(hidden_state, weights[step]) + biases[step])
            reg_l1 += tf.reduce_sum(tf.abs(weights[step]))
            reg_l2 += tf.nn.l2_loss(weights[step])

        return hidden_state, reg_l1, reg_l2

    def add_loss_op(self):
        loss = tf.nn.l2_loss(self.reward_placeholder \
                             + (self.vfa_config.discount ** self.vfa_config.nb_steps) * self.max_Q_targets \
                                - self.Q_inputs) \
               + self.vfa_config.alpha_reg * self.reg_l1 \
               + self.vfa_config.beta_reg * self.reg_l2
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return total_loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.vfa_config.lr)
        train_op = optimizer.minimize(loss, var_list=self.input_weights + self.input_biases)
        return train_op
    
    def run_batch(self, session, inputs, actions, outputs, terminal_states, rewards):
        feed = {self.input_placeholder: inputs,
                self.action_placeholder: actions,
                self.output_placeholder: outputs,
                self.terminal_placeholder: terminal_states,
                self.reward_placeholder: rewards}
        session.run(self.train_step, feed_dict=feed)

    def update_target_weights(self, session):
        for update in self.updates:
            session.run(update) 

    def best_next_actions(self, session, inputs):
        feed = {self.input_placeholder: inputs}
        next_actions = session.run(self.next_actions, feed_dict=feed)
        return next_actions

    def value_function(self, session, inputs, actions):
        feed = {self.input_placeholder: inputs,
                self.action_placeholder: actions}
        value_function = session.run(self.Q_inputs, feed_dict=feed)
        return value_function

