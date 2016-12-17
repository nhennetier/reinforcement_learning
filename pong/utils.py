import tensorflow as tf

class VFAConfig(object):
    height_dim = 84
    width_dim = 84
    action_space = [0, 1, 2, 3, 4, 5]
    lr = 1e-3
    discount = .99
    alpha_reg = 1e-4
    beta_reg = 1e-4
    nb_steps = 4

class ValueFunctionApproximation():
    '''Base class for value Function Approximation. Shouldn't be used separately.'''
    def __init__(self, vfa_config):
        self.vfa_config = vfa_config
        
        self.add_placeholders()
        self.add_variables()
        self.Q_inputs = self.add_comp_graph(self.input_placeholder)
        self.next_actions = tf.argmax(self.Q_inputs, 1)
        self.max_Q_inputs = tf.reduce_max(self.Q_inputs, 1)
        self.max_Q_inputs = tf.mul(self.max_Q_inputs, self.terminal_placeholder)
        self.Q_inputs = tf.reduce_sum(tf.mul(self.Q_inputs, self.action_placeholder), 1)
        
        self.calculate_loss = self.add_loss_op(self.Q_inputs)
        self.train_step = self.add_training_op(self.calculate_loss)

    def add_placeholders(self):
        #Current state
        self.input_placeholder = tf.placeholder(tf.float32,
            shape=[None, self.vfa_config.height_dim, self.vfa_config.width_dim, 3],
            name='Input')
        #Current action
        self.action_placeholder = tf.placeholder(tf.float32,
            shape=[None, len(self.vfa_config.action_space)],
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

    def add_variables(self):
        self.weights = {
            # 8x8 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([8, 8, 3, 32])),
            # 4x4 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([4, 4, 32, 64])),
            # 3x3 conv, 64 inputs, 64 outputs
            'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
            # fully connected, 3*3*64 inputs, 512 outputs
            'wd1': tf.Variable(tf.random_normal([11*11*64, 512])),
            # 512 inputs
            'out': tf.Variable(tf.random_normal([512, len(self.vfa_config.action_space)]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bc3': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([512])),
            'out': tf.Variable(tf.random_normal([len(self.vfa_config.action_space)]))
        }

    def conv2d(self, inputs, W, b, strides=1):
        x = tf.nn.conv2d(inputs, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def fc2d(self, inputs, W, b):
        fc1 = tf.reshape(inputs, [-1, W.get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, W), b)
        return tf.nn.relu(fc1)

    def add_comp_graph(self, inputs):
        conv1 = self.conv2d(inputs, self.weights['wc1'], self.biases['bc1'], strides=4)
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'], strides=2)
        conv3 = self.conv2d(conv2, self.weights['wc3'], self.biases['bc3'])
        
        fc1 = self.fc2d(conv3, self.weights['wd1'], self.biases['bd1'])
        # fc1 = tf.nn.dropout(fc1, self.dropout_placeholder)
        
        output = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return output

    def add_loss_op(self, inputs):
        loss = tf.nn.l2_loss(self.reward_placeholder \
                             + (self.vfa_config.discount ** self.vfa_config.nb_steps) * self.output_placeholder \
                                - inputs)
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

