import cnn
import numpy as np
from show import show

# =============================================
# (1) Instantiate an inference object
# =============================================
app_context = {
    'db': 'mr_prostate',
    'ip_mongodb': '160.87.25.45',
    'tags-series': ['pirads-t2'],
    'tags-labels': ['prostate']
}

inference = cnn.Inference(
    path='./final',
    app_context=app_context)

# =============================================
# (2) Find the studyid(s) (or doc(s)) that you wish to run through the network 
# =============================================
man = cnn.db.Manager(app_context['ip_mongodb'])
man.use('mr_prostate')

# (A) Find a random document
doc = man.find_one()

# (B) Find a document with studyid
doc = man.find_one({'study.studyid': 'MR-16-0008646'})

# (C) Find all valid cases (if fold == 0, for example)
docs = man.collection.find({'study.valid': {'$in': [0]}})
studyids = [d['study']['studyid'] for d in docs]

# =============================================
# (3) Use inference.run_single() 
# =============================================
# fileid = '5b8eb00906eac21e57ebc8ca'
fileid = '5b8eafd206eac21e57ebc8aa'

# outputs = inference.run_single(studyid=studyids[0])
outputs = inference.run_single(fileid=fileid)

argmax = np.argmax(outputs['logits']['dsc-pros'], axis=-1)
argmax = np.pad(argmax[0], [(3, 3), (0, 0), (0, 0)], mode='minimum')

show({'dat': outputs['dat'][0], 'lbl': argmax.astype('uint8')})
