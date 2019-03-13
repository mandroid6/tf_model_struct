import tensorflow as tf
from base.base_model import BaseModel


class Model(BaseModel):
    
    def __init__(self, config):
        super(Model, self).__init__(config)
        
        self.build_model()
        self.init_saver()
        
    
    def build_model(self):
        pass
    
    def init_saver(self):
        pass