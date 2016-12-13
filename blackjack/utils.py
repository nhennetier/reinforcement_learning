import tensorflow as tf

class VFAConfig(object):
    hidden_size = 100
    num_hidden_layers = 10
    state_dim = 560
    action_space = [0, 1]
    lr = 1e-2
    discount = 1/(1+1e-1)

class ValueFunctionApproximation():
    '''Base class for value Function Approximation. Shouldn't be used separately.'''
    def __init__(self, vfa_config):
        self.vfa_config = vfa_config
        
        self.add_placeholders()
        self.Q_inputs = self.add_comp_graph(self.input_placeholder)
        self.next_actions = tf.argmax(self.Q_inputs, 1)
        self.max_Q_inputs = tf.reduce_max(self.Q_inputs, 1)
        self.max_Q_inputs = tf.mul(self.max_Q_inputs, self.terminal_placeholder)
        self.Q_inputs = tf.gather(tf.reshape(self.Q_inputs, [-1]), self.action_placeholder)
        
        self.calculate_loss = self.add_loss_op(self.Q_inputs)
        self.train_step = self.add_training_op(self.calculate_loss)

    def add_placeholders(self):
        #Current state
        self.input_placeholder = tf.placeholder(tf.float32,
            shape=[None, self.vfa_config.state_dim],
            name='Input')
        #Current action
        self.action_placeholder = tf.placeholder(tf.int32,
            shape=[None],
            name='Action')
        #Next state
        self.output_placeholder = tf.placeholder(tf.float32,
            shape=[None],
            name='Output')
        #Dummy variables, whether the next state is terminal
        self.terminal_placeholder = tf.placeholder(tf.float32,
            shape=[None],
            name='Terminal')
        #Next reward
        self.reward_placeholder = tf.placeholder(tf.float32,
            shape=[None],
            name='Reward')

    def add_comp_graph(self, inputs):
        raise NotImplementedError

    def add_loss_op(self, inputs):
        loss = tf.nn.l2_loss(self.reward_placeholder + self.vfa_config.discount * self.output_placeholder - inputs)
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return total_loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.vfa_config.lr)
        train_op = optimizer.minimize(loss)
        return train_op
    
    def run_batch(self, session, inputs, actions, outputs, terminal_states, rewards):
        feed = {self.input_placeholder: outputs,
                self.terminal_placeholder: terminal_states}
        Q_outputs = session.run(self.max_Q_inputs, feed_dict=feed)
        
        feed = {self.input_placeholder: inputs,
                self.action_placeholder: actions,
                self.output_placeholder: Q_outputs,
                self.reward_placeholder: rewards}
        session.run(self.train_step, feed_dict=feed)

    def best_next_actions(self, session, inputs):
        feed = {self.input_placeholder: inputs}
        next_actions = session.run(self.next_actions, feed_dict=feed)
        return next_actions

    def value_function(self, session, inputs, actions):
        feed = {self.input_placeholder: inputs,
                self.action_placeholder: actions}
        value_function = session.run(self.Q_inputs, feed_dict=feed)
        return value_function


class LinearVFA(ValueFunctionApproximation):
    def add_comp_graph(self, inputs):
        with tf.variable_scope('Linear-Layer'):
            W = tf.get_variable(
                'W',
                [self.vfa_config.state_dim, len(self.vfa_config.action_space)],
                dtype=tf.float32)
            output = tf.matmul(inputs, W)
        return output


class NeuralNetworkVFA(ValueFunctionApproximation):
    def add_comp_graph(self, inputs):
        with tf.variable_scope('Input-Layer'):
            W = tf.get_variable(
                'W',
                [self.vfa_config.state_dim, self.vfa_config.hidden_size],
                dtype=tf.float32)
            b = tf.get_variable(
                'b',
                [self.vfa_config.hidden_size],
                dtype=tf.float32)
            hidden_state = tf.nn.tanh(tf.matmul(inputs, W) + b)

        for step in range(self.vfa_config.num_hidden_layers+1):
            with tf.variable_scope('Layer-%s' % (step+1)):
                W = tf.get_variable(
                    'W',
                    [self.vfa_config.hidden_size, self.vfa_config.hidden_size],
                    dtype=tf.float32)
                b = tf.get_variable(
                    'b',
                    [self.vfa_config.hidden_size],
                    dtype=tf.float32)
            hidden_state = tf.nn.tanh(tf.matmul(hidden_state, W) + b)
        
        with tf.variable_scope('Projection-Layer'):
            W = tf.get_variable(
                'W',
                [self.vfa_config.hidden_size, len(self.vfa_config.action_space)],
                dtype=tf.float32)
            b = tf.get_variable(
                'b',
                [len(self.vfa_config.action_space)],
                dtype=tf.float32)
            output = tf.matmul(hidden_state, W) + b

        return output

