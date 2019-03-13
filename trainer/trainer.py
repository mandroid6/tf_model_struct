from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(Trainer, self).__int__(sess, model, data, config, logger)
        
        
    def train_epoch(self):
        # logic for each epoch
        pass
    
    def train_step(self):
        # implement logic for train step
        pass
    
    