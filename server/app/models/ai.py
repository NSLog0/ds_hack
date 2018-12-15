import os.path
import pickle
import pandas as pd
import numpy as np
import json

class Predictor:
    def predict(data):
        rpath = os.path.abspath(os.path.dirname(__file__))
        ss = os.path.join(rpath, '../train_models/ss.pkl')
        clf = os.path.join(rpath, '../train_models/clf.pkl')

        clf = pickle.load(open(clf, "rb"))
        ss = pickle.load(open(ss, "rb"))

        # pp = os.path.join(rpath, '../train_models/data/RFT/Test1.csv')
        # df_test1 = pd.read_csv(pp)

        unique_fluid_types = ['O', 'W', 'G', 'O?']
        unique_subblocks = ['F', 'N', 'P', 'C', 'S', 'K', 'R', 'Q', 'D', 'J', 'G', 'O', 'E', 'L', 'M', 'I', 'H', 'B']

        features = ['DepthMD',
                    'DepthTVDSS',
                    'Temp',
                    'GR',
                    'Resist_deep',
                    'Resist_medium',
                    'Resist_short',
                    'Density',
                    'Neutron',
                    'FluidType',
                    'Subblock',
                    'Thickness',]

        # sample = df_test1.iloc[0:1]
        # print(sample)

        sample = pd.io.json.json_normalize(json.loads(data))
        sample = sample[features]

        lst_fluid_type = []
        lst_subblock = []

        train_fluid_type = np.zeros(len(unique_fluid_types))
        train_subblock = np.zeros(len(unique_subblocks))
        this_fluid_type = sample['FluidType'].values[0][0]
        this_fluid_type_index = unique_fluid_types.index(this_fluid_type)
        this_subblock = sample['Subblock'].values[0][0]
        this_subblock_index = unique_subblocks.index(this_subblock)
        train_fluid_type[this_fluid_type_index] = 1
        train_subblock[this_subblock_index] = 1
        lst_fluid_type.append(train_fluid_type)
        lst_subblock.append(train_subblock)

        lst_fluid_type = np.asarray(lst_fluid_type)
        lst_subblock = np.asarray(lst_subblock)
        df_fluid_type = pd.DataFrame(lst_fluid_type)
        df_subblock = pd.DataFrame(lst_subblock)

        X_train1 = sample.reset_index(drop=True).drop(columns=['FluidType', 'Subblock'])
        df_fluid_type = df_fluid_type.reset_index(drop=True)
        df_subblock = df_subblock.reset_index(drop=True)
        df_one_hot_encoded = pd.concat([X_train1, df_fluid_type, df_subblock], axis=1, ignore_index=True)

        to_predict = ss.transform(df_one_hot_encoded)
        pred = clf.predict(to_predict)
        print(pred)

        return pred
