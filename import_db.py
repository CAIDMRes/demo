import cnn
import glob, os, pickle
import numpy as np

# --- Global variables
modalities = ['t2', 'flair', 't1pre', 't1post']

app_context = {
    'name': 'brats',
    'db': 'brats',
    'dst_root': '/data/mvk-mzl'}

infos = {
    'shape': [155, 240, 240],
    'tiles': [0, 0, 0]}

def convert_files():
    """
    Method to convert files from raw memory mapped to regular Numpy

    """
    # --- Make output directory
    os.makedirs('/data/raw/brats/data/brats/converted', exist_ok=True)

    fnames = glob.glob('/data/raw/brats/data/brats/npy/*/dat.npy')
    for n, fname in enumerate(fnames):

        print('Converting file %04i / %04i' % (n + 1, len(fnames)), end='\r')

        # --- Convert data
        dat = load_mmap_npy(fname, dtype='int16')
        sid = fname.split('/')[-2]
        dst_root = '/data/raw/brats/data/brats/converted/%s' % sid
        os.makedirs(dst_root, exist_ok=True)

        for ch, m in enumerate(modalities):
            dst = '%s/%s.npy' % (dst_root, m)
            np.save(dst, dat[..., ch:ch+1])

        # --- Convert labels
        lbl = load_mmap_npy(fname.replace('dat.npy', 'lbl.npy'), dtype='uint8')
        dst = '%s/lbl.npy' % dst_root
        np.save(dst, lbl)

def load_mmap_npy(fname, dtype):
    
    dat = np.memmap(fname, dtype=dtype, mode='r')
    dat = dat.reshape(155, 240, 240, -1)
    
    return dat

def create_documents():

    roots = glob.glob('/data/raw/brats/data/brats/converted/*/')
    studyid = lambda x : x.split('/')[-2]

    documents = []
    for root in roots:

        # --- Make base document
        document = {
            'study': {'studyid': studyid(root)},
            'series': [],
            'labels': [{
                'file': '%slbl.npy' % root,
                'type': 'orig',
                'tags': ['tumor']
            }]}

        # --- Add all series
        for modality in modalities:
            data = {
                'data': [{
                    'file': '%s%s.npy' % (root, modality),
                    'type': 'orig',
                    'tags': ['brats-all', 'brats-%s' % modality]
                    }]}
            document['series'].append(data)

        documents.append(document)

    os.makedirs('./pkls', exist_ok=True)
    pickle.dump(documents, open('./pkls/documents_all.pkl', 'wb'))

def run():

    # --- Initialize importer
    importer = cnn.db.Importer(app_context)

    # --- Load documents
    documents = pickle.load(open('./pkls/documents_all.pkl', 'rb'))
    importer.documents = documents[5:]

    # --- Import
    importer.run(infos=infos)
    importer.validation_split(opts={'folds': 5})
    

if __name__ == '__main__':

    # =========================================================
    # (1) Convert files from memory-mapped to regular
    # =========================================================
    # convert_files()

    # =========================================================
    # (2) Create documents pickle file 
    # =========================================================
    # create_documents()

    # =========================================================
    # (3) Import 
    # =========================================================
    run()
