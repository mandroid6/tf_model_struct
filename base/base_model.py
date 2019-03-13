import tensorflow as tf

# this class needs to be inherited by all model classes

# ensures that below methods are overided by every model
class BaseModel:
	def __init__(self):
		pass
		
	def save(self):
		pass	
		
	def load(self):
		pass
		
	def init_current_epoch(self):
		pass
		
	def init_global_step(self):
		pass
    
    def init_saver(self):
        pass
    
    def build_model(self):
        pass
		
	