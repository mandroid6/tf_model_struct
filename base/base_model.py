import tensorflow as tf

# this class needs to be inherited by all model classes

# ensures that below methods are overided by every model
class BaseModel:
	def __init__(self, config):
		self.config = config
        self.init_global_step()
        self.init_current_epoch()
        
		
	def save(self, sess):
		print('Saving Model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print('Model Saved')
		
	def load(self, sess):
		latest_ckpt = tf.train.latest_checkpoint(self.config.checkpoint_dir)
		if latest_ckpt:
            print('Loading model checkpoint {} ...\n'.format(latest_ckpt))
            self.saver.restore(sess, latest_ckpt)
            print('Model Loaded')
            
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
		
	