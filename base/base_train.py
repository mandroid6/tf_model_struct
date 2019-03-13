import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        
        self.sess = sess
        self.data = data
        self.config = config
        self.model = model
        self.logger = logger
        
        
        # global and local variable initializers
        self.init = tf.group(tf.global_variable_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
    
    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
    
    def train_epoch(self):
        # logic for epoch
        # loop over no of iterations in config and call train step
        
        raise NotImplementedError
    
    def train_step(self):
        # logic for train step
        # run tf.session
        raise NotImplementedError