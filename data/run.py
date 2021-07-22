#%%
import os
# from tensorflow import keras
from keras.models import model_from_json;



#%%
def company(paths):
    with open(paths['model'],'r') as json_file:
        model = model_from_json(json_file.read())
        model.load_weights(paths["weight"])
        print('weights-loaded')
        model.summary()  

root_path="D:\MSC-Project\company-data\CompanyData-AEL\GRU[32, 32, 32][32, 32]\(14, None, 8, 0.0001, 0.8, 0.99, 3.3)\model"
paths = {}
paths['model'] = os.path.join(root_path , "model.json" )
paths["weight"] =os.path.join(root_path , "weights.hdf5" )

company(paths)





# %%
