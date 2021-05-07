import pandas as pd
import numpy as np

def matching_id(data_df,label_df, q_id):
    valid_idx = ~pd.isnull(label_df.loc[q_id,:])
    label_df = label_df.loc[q_id, valid_idx]

    data_id = data_df.columns
    label_id = label_df.index

    valid_data, valid_label = [], []
    for i, id_ in enumerate(data_id):
        if id_ in label_id:
            valid_data.append(data_df[id_])
            valid_label.append(label_df[id_])
    valid_data = pd.concat(valid_data, axis=1)
    valid_label = np.array(valid_label)
    return valid_data, valid_label