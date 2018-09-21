import os
import cnn

from show import show
import numpy as np

class Client(cnn.utils.Client):
    
    def init_client(self):
        """
        Method to set default experiment specific variables
        
        """
        self.infos = {
            'shape': [5, 256, 256],
            'tiles': [2, 0, 0],
            'valid_expansion': True
        }
        
        self.inputs = {
            'dtypes': {},
            'shapes': {},
            'classes': {}
        }
        
        self.inputs['dtypes']['dat'] = 'float32'
        self.inputs['shapes']['dat'] = [5, 256, 256, 2] # pre+2 post
        
        self.inputs['dtypes']['sce-tumor'] = 'int32'
        self.inputs['shapes']['sce-tumor'] = [1, 256, 256, 1]
        self.inputs['classes']['sce-tumor'] = 2 #leisions 
        
        self.dist = {
            0: 0.5, 
            1: 0.5}
        
        self.mode = 'mixed'
        self.iids = {}
        self.x = {}
        
    def get(self, mode=None, random=True):
        """
        Method to load a single train/valid study
        
        """
        # Load data
        next_doc = self.next_doc(mode=mode, random=True)
        vol = self.load(doc=next_doc['doc'], infos=next_doc['infos'])
        
        # Preprocessing lbl
        # ========================================================
        # HOW TO MASK A LOSS FUNCTION
        # ========================================================
        pp = next_doc['doc']['series'][0]['data'][0]['pp']
        vol['lbl'] += 1
        vol['lbl'][vol['dat'][..., 1:] < pp['mean']] = 0

        # Preprocessing dat 
        # ========================================================
        # CONSIDER MEAN/SD FOR ENTIRE VOLUME FROM next_doc['doc'] 
        # ========================================================
        vol['dat'] = (vol['dat'] - pp['mean']) / pp['std'] 

        # Create nested vol dictionary
        vol['lbl'] = {'sce-tumor': vol['lbl'][2:-2]}
        
        return self.return_get(next_doc, vol)

class Model(cnn.SemanticModel):

   def init_hyperparams_custom(self):

       self.params['batch_size'] = 8 
       self.params['iterations'] = 20000
       self.params['l2_beta'] = 0 

       self.params['enet_fn'] = self.create_enet_vgg
       self.params['train_ratio'] = {'enet': 1}
       self.params['learning_rate'] = {'enet': 1e-3}

        # ========================================================
        # CONSIDER WEIGHTING LOSS FUNCTION FOR CLASS IMBALANCE 
        # ========================================================
       self.params['class_weights'] = {'sce-tumor': {1: 1, 2: 5}}

       self.params['stats_matrix_shape'] = [60, None, None]
       self.params['stats_mode'] = 'agg'
       self.params['stats_top_model_source'] = {
           'name': 'enet',
           'node': 'errors',
           'target': 'dsc-tumor',
           'key': 1} #1 = leisions? 0 = whole breast

   def create_enet_vgg(self, inputs):

       nn_struct = {}

       # ------------------------------------------------
       # | FEATURE MAP DIMENSIONS|
       # L0: 256 
       # L1: 128-128-128
       # L2: 064-064-064
       # L3: 032-032-032
       # L4: 016-016-016
       # L5: 008-008

       nn_struct['channels_out'] = [4,
           16, 16, 16,
           32, 32, 32,
           64, 64, 64,
           96, 96, 96,
           128, 128]

       nn_struct['filter_size'] = [[1, 3, 3],
           [1, 3, 3], [2, 1, 1], [1, 3, 3], 
           [1, 3, 3], [2, 1, 1], [1, 3, 3], 
           [1, 3, 3], [2, 1, 1], [1, 3, 3], 
           [1, 3, 3], [2, 1, 1], [1, 3, 3], 
           [1, 3, 3], [1, 3, 3]]
       
       nn_struct['padding'] = ['SAME',
           'SAME', 'VALID','SAME',
           'SAME', 'VALID','SAME',
           'SAME', 'VALID','SAME',
           'SAME', 'VALID','SAME',
           'SAME','SAME']
       
       
       
       nn_struct['stride'] = [1,
           [1, 2, 2], 1,1,
           [1, 2, 2], 1,1,
           [1, 2, 2], 1,1,
           [1, 2, 2], 1,1,
           [1, 2, 2], 1]
       
       nn_struct['decoder'] = True
       nn_struct['decoder_hybrid'] = True

       self.builder.create(
           name='E',
           nn_struct=nn_struct,
           input_layer=inputs[0])

       self.create_logits(name='E')

app_context = {
    'db': 'mr_breast',
    'ip_mongodb': '160.87.25.45',
    'tags-series': ['dce_pre-toy', 'dce_post01-toy'],
    'tags-labels': ['tumor']
}

# ====================================================
# RUN NETWORK TRAINING
# ====================================================
# net = cnn.Network()
# net.Client = Client
# net.Model = Model
# net.Stats = cnn.SemanticStats
#
# net.initialize(
#     save_dir='./exp02',
#     app_context=app_context, 
#     fold=-1,
#     threads=2, 
#     batch=8,
#     capacity=128
# )
#
# net.train()

# ====================================================
# TESTING CLIENT 
# ====================================================
# client = Client(app_context)
# client.prepare(fold=-1)
#
# dats = []
# lbls = []
# for i in range(32):
#     print(i)
#     it = client.get()
#     it = client.create_inputs_dict(it, mode='train')
#     dats.append(it['dat'][2, ..., 1:])
#     lbls.append(it['lbl']['sce-tumor'][0])
#
# dats = np.stack(dats, axis=0)
# lbls = np.stack(lbls, axis=0)
