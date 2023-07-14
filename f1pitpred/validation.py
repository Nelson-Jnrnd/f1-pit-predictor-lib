import os
import pickle
import numpy as np
from tensorflow.keras import models

class model:
    def __init__(self, model_name, model_type, model_path):
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path

    def load_model(self):
        if self.model_type == 'sklearn':
            self.model = pickle.load(open(self.model_path, 'rb'))
        elif self.model_type == 'keras':
            self.model = models.load_model(self.model_path, compile=False)
        else:
            raise ValueError('Model type not supported')
    
    def load_encoder(self, encoder_path):
        self.encoder = pickle.load(open(encoder_path, 'rb'))

    def predict(self, X):
        X = np.array(X).astype(np.float32)
        if self.model_type == 'sklearn':
            return self.model.predict(X)
        elif self.model_type == 'keras':
            return self.model.predict(X).round()
        else:
            raise ValueError('Model type not supported')
    
    def predict_proba(self, X):
        X = np.array(X).astype(np.float32)
        if self.model_type == 'sklearn':
            return [y[1] for y in self.model.predict_proba(X)]
        elif self.model_type == 'keras':
            return self.model.predict(X)
        else:
            raise ValueError('Model type not supported')