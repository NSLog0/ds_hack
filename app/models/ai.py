import os.path
import pickle
import pandas as pd
import numpy as np
import json

class Predictor:
    def predict(data):
        rpath = os.path.abspath(os.path.dirname(__file__))
        ss = os.path.join(rpath, '../train_models/new/ss.pkl')
        clf = os.path.join(rpath, '../train_models/new/clf.pkl')
        fluid_clf = os.path.join(rpath, '../train_models/new/fluid_type_clf.pkl')
        reg = os.path.join(rpath, '../train_models/new/reg.pkl')

        clf = pickle.load(open(clf, "rb"))
        ss = pickle.load(open(ss, "rb"))
        reg = pickle.load(open(reg, 'rb'))
        fluid_clf = pickle.load(open(fluid_clf, 'rb'))

        # pp = os.path.join(rpath, '../train_models/data/RFT/Test1.csv')
        # df_test1 = pd.read_csv(pp)

        features = ['DepthMD',
                    'DepthTVDSS',
                    'Temp',
                    'GR',
                    'Resist_deep',
                    'Resist_medium',
                    'Resist_short',
                    'Density',
                    'Neutron',
                    'Thickness']

        sample = pd.io.json.json_normalize(json.loads(data))
        sample = sample[features]

        reg_sample = sample[['Resist_deep', 'Resist_medium', 'Resist_short', 'Density', 'Neutron', 'GR']]

        sample = np.asarray(sample)
        reg_sample = np.asarray(reg_sample)
        x_test = ss.transform(sample)

        # predict fluid type
        fluid_type = fluid_clf.predict(x_test)[0] # 1: Oil, 0: Not Oil
        is_normal = clf.predict(x_test)[0] #1: NORMAL, 0: OTHER
        mobility_score = reg.predict(reg_sample)[0] # Mobility score

        return (fluid_type, is_normal, mobility_score)
