import tensorflow as tf

# this class needs to be inherited by all model classes

# ensures that below methods are overided by every model
class BaseModel:
	def __init__(self):
		self.config = config
        self.init_global_step()
        self.init_current_epoch()
        
		
	def save(self):
		pass	
		
	def load(self):
		pass
		
	def init_current_epoch(self):
		with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)
		
	def init_global_step(self):
		with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        
    
    def init_saver(self):
        # this needs to be implemented in child class
        raise NotImplementedError
    
    def build_model(self):
        # this needs to be implemented in child class
        raise NotImplementedError
		
	