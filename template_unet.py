import cnn
import numpy as np
from show import show

app_context = {
    'db': 'brats',
    'tags-series': ['brats-t2'],
    'tags-labels': ['tumor']
}

# ==========================================================================================
# U-NET 
# ==========================================================================================

fdir = './exps/unet/exp01'

class Client(cnn.utils.Client):

    def init_client(self):
        """
        Method to set default experiment specific variables 

        """
        self.infos = {
            'shape': [5, 256, 256],
            'tiles': [5, 0, 0],
            'affine': True,
            'translation': [-10, 10],
            'scale': [0.8, 1.2],
            'rotation': [-0.78, 0.78],
            'shear': [-0.1, 0.1],
            'valid_padding': [2, 0, 0],
            'valid_expansion': True}

        self.inputs = {}

        self.inputs['dtypes'] = {
            'dat': 'float32',
            'dsc-tumor': 'int32'}

        self.inputs['shapes'] = {
            'dat': [5, 256, 256, 1],
            'dsc-tumor': [1, 256, 256, 1]}

        self.inputs['classes'] = {
            'dsc-tumor': 5}

        # self.dist = {}
        self.x = {}

        self.mode = 'mixed'

    def get(self, mode=None, random=True, vol=None, studyid=None, fileid=None, doc=None, infos=None):
        """
        Method to load a single train/valid study with preprocessing

        """
        # Load data
        if vol is None:
            next_doc = self.next_doc(mode=mode, random=random, doc=doc, studyid=studyid, fileid=fileid, infos=infos)
            vol = self.load(doc=next_doc['doc'], infos=next_doc['infos'])
        else:
            vol, next_doc = self.init_vol(vol, mode)
            vol['dat'] = np.pad(vol['dat'], ((2, 2), (0,0), (0,0), (0,0)), mode='constant')

        z = np.arange(2, vol['dat'].shape[0] - 2)

        # Preprocessing
        if vol['dat'].any():
            vol['dat'] = (vol['dat'] - np.mean(vol['dat'])) / np.std(vol['dat'])
        else:
            vol['dat'][:] = -1

        # Create nested vol dictionary
        vol['lbl'] = {'dsc-tumor': vol['lbl'][z] + 1} 

        return self.return_get(next_doc, vol)

    def reverse_normalization(self, dat, doc=None):
        """
        Method to reverse normalization

        """
        return (dat * 120).astype('int16')

class Model(cnn.SemanticModel):

    def init_hyperparams_custom(self):

        self.params['batch_size'] = 8 
        self.params['iterations'] = 1000 

        self.params['enet_fn'] = self.create_enet_unet
        self.params['train_ratio'] = {'enet': 1}
        self.params['learning_rate'] = {'enet': 1e-4}

        self.params['stats_matrix_shape'] = [32, None, None]
        self.params['stats_logits'] = [] 
        self.params['stats_mode'] = 'agg'

        self.params['print_width_A'] = 9
        self.params['print_width_B'] = 7 

        self.params['stats_top_model_source'] = {
            'name': 'enet',
            'node': 'errors',
            'target': 'dsc-tumor',
            'key': 1}

    def create_enet_unet(self, inputs):

        nn_struct = {}

        # ------------------------------------------------
        # L1: 256-256
        # L2: 128-128 
        # L3: 064-064 
        # L4: 032-032
        # L5: 016-016
        # L5: 008-008

        nn_struct['channels_out'] = [4,
            16, 16,
            32, 32,
            48, 48, 48,
            48, 48, 48,
            64, 64, 48,
            64, 64, 64
        ]

        nn_struct['filter_size'] = [[1, 7, 7],
            [1, 3, 3], [1, 3, 3],
            [1, 3, 3], [1, 3, 3],
            [1, 3, 3], [2, 1, 1], [1, 3, 3],
            [1, 3, 3], [2, 1, 1], [1, 3, 3],
            [1, 3, 3], [2, 1, 1], [1, 3, 3],
            [1, 3, 3], [2, 1, 1], [1, 3, 3]
        ]
        
        nn_struct['stride'] = [1,
            [1, 2, 2], 1,
            [1, 2, 2], 1,
            [1, 2, 2], 1, 1,
            [1, 2, 2], 1, 1,
            [1, 2, 2], 1, 1,
            [1, 2, 2], 1, 1
        ]

        nn_struct['padding'] = ['SAME',
            'SAME', 'SAME',
            'SAME', 'SAME',
            'SAME', 'VALID', 'SAME',
            'SAME', 'VALID', 'SAME',
            'SAME', 'VALID', 'SAME',
            'SAME', 'VALID', 'SAME',
        ] 
        
        nn_struct['decoder'] = True
        nn_struct['decoder_hybrid'] = True

        self.builder.create(
            name='E',
            nn_struct=nn_struct,
            input_layer=inputs[0])

        self.create_logits(name='E')

def run_inference():

    pass

if __name__ == '__main__':

    # ======================================================
    # TEST TRAINING
    # ======================================================
    # net = cnn.Network()
    # net.Client = Client
    # net.Model = Model
    # net.Stats = cnn.SemanticStats
    #
    # net.initialize(
    #     save_dir=fdir,
    #     app_context=app_context, 
    #     fold=-1, 
    #     threads=4, 
    #     batch=16,
    #     capacity=128
    # )
    # net.train()

    # ======================================================
    # TEST CLIENT 
    # ======================================================

    # run_inference()
    
    # ======================================================
    # TEST CLIENT 
    # ======================================================
    # client = Client(
    #     app_context=app_context, 
    #     # threads=2,
    #     # batch=16,
    #     # capacity=128
    # )
    # client.prepare(fold=0)
    #
    # import time
    # st = time.time()
    # sids = {}
    # for i in range(20):
    #     batch = client.get()
    #     # batch = client.get_async()
    #     batch = client.create_inputs_dict(batch, mode='train')
    #     # print('%04i: single example loaded' % i)
    #
    # print('Total elapsed time: %4.4f sec' % (time.time() - st))

    pass
