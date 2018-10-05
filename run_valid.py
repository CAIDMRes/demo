import cnn
import numpy as np, pickle
from show import show

# =============================================
# (1) Instantiate an inference object
# =============================================
app_context = {
    'db': 'mr_breast',
    'tags-series': ['dce_pre', 'dce_post01'],
    'tags-labels': ['tumor']
}

# inference = cnn.Inference(
#     path='/home/turkay/mri-breast/Test/exps/unets/exp4/final',
#     app_context=app_context)

# =============================================
# (2a) Find the studyid(s) (or doc(s)) that you wish to run through the network 
# =============================================
man = cnn.db.Manager()
man.use('mr_prostate')

# (A) Find a random document
doc = man.find_one()

# (B) Find a document with studyid
doc = man.find_one({'study.studyid': 'MR-16-0008646'})

# (C) Find all valid cases (if fold == 0, for example)
docs = man.collection.find({'study.valid': {'$in': [0]}})
studyids = [d['study']['studyid'] for d in docs]

# =============================================
# (2b) Find the fileid(s) that you wish t o run through the network 
# =============================================

# =============================================
# STATS 
# =============================================
# 
# The 'matrix' entry in stats contains a cumulative summary of training statistics.
# 
# The tree structure is as follows:
#
#   >> stats['matrix'][net][node][target][key]
# 
# net = network name ('enet', 'lnet', etc)
# node = 'losses' or 'errors'
# target = name(s) of the loss/error names (created in Client, e.g. 'dsc-xxx', 'sce-xxx', etc)
# key = if node == 'errors', a key must also be provided (e.g. 0, 1, 2 for dsc, 'topk' for sce, etc)
# 
# =============================================

# stats['matrix']

# Open Stats object
pickle_file = '/home/turkay/mri-breast/Test/exps/unets/exp4/stats.pkl'
stats = pickle.load(open(pickle_file, 'rb'))

# How to calculate Dice per patient
matrix = stats['matrix']['enet']['errors']['dsc-tumor'][1]
matrix = np.nansum(matrix, axis=(1,2,3))

dice = matrix[:, 0] / matrix[:, 1]

# Find fileids
fileids = stats['fileids']['all']

# Make dict for errors {'fileid': dice}
train = dict([(str(f['_id']), d) for f, d in zip(fileids['train'], dice[:len(fileids['train'])])])
valid = dict([(str(f['_id']), d) for f, d in zip(fileids['valid'], dice[-len(fileids['valid']):])])

# =============================================
# (3) Use inference.run_single() 
# =============================================
# fileid = '5b8eb00906eac21e57ebc8ca'
# fileid = '5b8eafd206eac21e57ebc8aa'

# outputs = inference.run_single(studyid=studyids[0])
# outputs = inference.run_single(fileid=fileid)
#
# argmax = np.argmax(outputs['logits']['dsc-pros'], axis=-1)
# argmax = np.pad(argmax[0], [(3, 3), (0, 0), (0, 0)], mode='minimum')
#
# show({'dat': outputs['dat'][0], 'lbl': argmax.astype('uint8')})
