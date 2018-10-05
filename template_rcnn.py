import cnn
import numpy as np, bson
from show import show

app_context = {
    'db': 'brats',
    'tags-series': ['brats-flair'],
    'tags-labels': ['tumor']
}

# ==========================================================================================
# R-CNN |
# ==========================================================================================

fdir = '.'

rcnn = cnn.rcnn.RCNN(
    
    # --- Baseline considerations
    classes=['sce-tumor'],
    c=[4, 5, 6],
    scale_factor=2,
    original_shape=[256, 256],

    # --- Use for "object-like" abnormalities
    # convert_msk_func='default',
    # iou_upper=0.8, 
    # iou_lower=0.3,
    # gt_box_padding=0,

    # --- Use for more unusual shaped lesions
    convert_msk_func='pyramid',
    pyramid_min_overlap=0.3,
    pyramid_max_delta=0.5,
    pyramid_coverage=0.99,
    pyramid_nms=False,
    iou_upper=0.5, 
    iou_lower=0.3,
    gt_box_padding=0,

    # --- Head network (part II)
    image_centric=32,
    nms_top_N=32,
    nms_iou_threshold=0.7,
    nms_func='fast',
    save_dir=fdir)

class Client(cnn.rcnn.Client):

    def init_client(self):
        """
        Method to set default experiment specific variables 

        """
        if not hasattr(self, 'rcnn'): self.rcnn = rcnn

        self.infos = {
            'shape': [5, 256, 256],
            'tiles': [5, 0, 0],
            'valid_padding': [2, 0, 0],
            'valid_expansion': True}

        self.inputs = {}

        self.inputs['dtypes'] = {
            'dat': 'float32',
            'sce-tumor': 'int32'}

        N = self.rcnn.anchors['orig'].shape[0]

        self.inputs['shapes'] = {
            'dat': [5, 256, 256, 1],
            'sce-tumor': [1, N, 1, 1]}

        self.inputs['classes'] = {
            'sce-tumor': 2}

        self.dist = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 1.0}

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

        # Prepare lbl (all voxels == 2 are foreground, == 1 are background, == 0 are ignore)
        vol['lbl'][vol['lbl'] >= 1] = 2
        vol['lbl'][vol['lbl'] == 0] = 1 
        vol['lbl'][vol['dat'] == 0] = 0

        # Alternatively prepare mask containing lesion(s)
        msk = vol['lbl'] > 0

        # Preprocessing
        if vol['dat'].any():
            vol['dat'] = (vol['dat'] - np.mean(vol['dat'])) / np.std(vol['dat'])
        else:
            vol['dat'][:] = -1

        # Fill in empty labels if needed 
        if len(vol['lbl']) == 0:
            vol['lbl'] = np.zeros(vol['dat'].shape, dtype='uint8')
            vol['lbl'] = vol['lbl'][z] + msk[z] 

        # Prepare labels / masks
        vol = self.rcnn.prepare_lbl_msk(vol, msk, z, mode, self.inputs['shapes'])

        return self.return_get(next_doc, vol)

    def reverse_normalization(self, dat, doc=None):
        """
        Method to reverse normalization

        """
        return (dat * 120).astype('int16')

class Model(cnn.rcnn.Model):

    def init_hyperparams_custom(self):

        if not hasattr(self, 'rcnn'): self.rcnn = rcnn
        
        self.params['batch_size'] = 8 
        self.params['iterations'] = 200000 

        self.params['learning_rate'] = {'rpn': 1e-4, 'rcnn': 1e-4}
        self.params['train_ratio'] = {'rpn': 0, 'rcnn': 1}

        self.params['stats_matrix_shape'] = [32, None, None]
        self.params['stats_logits'] = ['sce-tumor', 'sce-scores', 'sl1-deltas_rpn', 'sl1-deltas_box', 'postnms'] 
        self.params['stats_mode'] = 'agg'

        self.params['stats_top_model_source'] = {
            'name': 'rcnn',
            'node': 'errors',
            'target': 'sce-tumor',
            'key': 'topk'}

    def create_enet_rcnn(self, inputs):

        nn_struct = {}

        # ------------------------------------------------
        # | FEATURE MAP DIMENSIONS |
        # C2: 128-128 
        # C3: 064-064 
        # C4: 032-032
        # C5: 016-016
        # C6: 008-008

        nn_struct['channels_out'] = [4,
            16, 16,
            32, 32, 32,
            48, 48, 48,
            64, 64, 64,
            64, 64, 64,
        ]

        nn_struct['filter_size'] = [[1, 7, 7],
            [1, 3, 3], [1, 3, 3],
            [1, 3, 3], [2, 1, 1], [1, 3, 3],
            [1, 3, 3], [2, 1, 1], [1, 3, 3],
            [1, 3, 3], [2, 1, 1], [1, 3, 3],
            [1, 3, 3], [2, 1, 1], [1, 3, 3],
        ]
        
        nn_struct['stride'] = [1,
            [1, 2, 2], 1,
            [1, 2, 2], 1, 1,
            [1, 2, 2], 1, 1,
            [1, 2, 2], 1, 1,
            [1, 2, 2], 1, 1,
        ]

        nn_struct['padding'] = ['SAME',
            'SAME', 'SAME',
            'SAME', 'VALID', 'SAME',
            'SAME', 'VALID', 'SAME',
            'SAME', 'VALID', 'SAME',
            'SAME', 'VALID', 'SAME'
        ] 
        
        nn_struct['rcnn'] = self.create_rcnn_struct()
        nn_struct['fpn'] = True

        # --- Create FPN
        self.builder.create(
            name='E',
            nn_struct=nn_struct,
            input_layer=inputs[0])

        # --- Create R-CNN head 
        self.finalize_rcnn(
            inputs=inputs,
            crop_size=[7, 7],
            fc_layers=[64])

def visualize_rcnn(n=9):
    """
    Method to visualize boxes for R-CNN architecture

    Use this method to titrate iou_upper/iou_lower

    Make sure to set image_centric == None

    """
    client = Client(app_context)
    client.prepare(fold=0)
    images = {0: [], 1: [], 2: []}
    vols = {0: {'dat': [], 'lbl': []}, 1: {'dat': [], 'lbl': []}, 2: {'dat': [], 'lbl': []}}

    for i in range(n):

        batch = client.get(mode='train')
        batch = client.create_inputs_dict(batch, mode='train')
        scores = np.squeeze(batch['lbl']['sce-scores'])

        # --- Modifiy batch['dat']
        if batch['dat'].shape[0] > 1:
            z = np.floor(batch['dat'].shape[0] / 2).astype('int')
            batch['dat'] = batch['dat'][z]

        print('%03i: IOU thresh %0.2f yields %03i labels >= 2' % 
            (i + 1, rcnn.params_rpn['iou_upper'], np.count_nonzero(scores >= 2)))

        for key in images:

            indices = np.nonzero(scores == key)[0]
            indices = indices[np.random.permutation(indices.size)][:10]

            # --- Apply deltas
            anchors_, _ = rcnn.apply_deltas(deltas=np.squeeze(batch['lbl']['sl1-deltas_rpn']))
            
            im = client.rcnn.visualize(
                im=np.squeeze(batch['dat']),
                anchors=client.rcnn.anchors['orig'],
                # anchors=anchors_,
                indices=indices,
                stroke=3,
                vm=[0, 0.66])

            images[key].append(im)

            boxes = np.expand_dims(im[..., 0] != im[..., 1], axis=-1)
            vols[key]['dat'].append(batch['dat'])
            vols[key]['lbl'].append(boxes.astype('uint8'))

    for key in [2, 1, 0]:

        vols[key]['dat'] = np.stack(vols[key]['dat'])
        vols[key]['lbl'] = np.stack(vols[key]['lbl'])
        show(vols[key]) 
        if input('Continue (y/n)? ') != 'y':
            break

# visualize_rcnn()

if __name__ == '__main__':

    # ======================================================
    # TEST TRAINING
    # ======================================================
    # net = cnn.Network()
    # net.Client = Client
    # net.Stats = cnn.rcnn.Stats
    # net.Model = Model
    #
    # net.initialize(
    #     save_dir=fdir,
    #     app_context=app_context, 
    #     fold=0, 
    #     # threads=2, 
    #     # batch=16,
    #     # capacity=128
    # )
    # net.train()

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
    # sids = [] 
    # for i in range(500):
    #     batch = client.get()
    #     # batch = client.get_async()
    #     batch = client.create_inputs_dict(batch, mode='train')
    #
    # print('Elapsed time: %2.5f sec' % (time.time() - st))

    pass
