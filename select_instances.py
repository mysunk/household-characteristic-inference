"""
Remove instances which could incur negative transfer
"""

from scipy.stats import norm
import numpy as np

from utils.metrics import KL


class SelectInstances:
    
    def __init__(self):
        pass

    def get_kl_result_removing_instances(self, source_dataset, target_dataset):
        """
        Removing instances one by one and calculate KL divergence between source and target dataset
        """
        
        kl_result = []
        n_data = len(source_dataset)
        for i in range(n_data):
            a = list(range(n_data))
            a.remove(i)
            source_data  = source_dataset.data[a,:]
            target_data = target_dataset

            mean_1, std_1 = np.mean(source_data), np.std(source_data)
            mean_2, std_2 = np.mean(target_data), np.std(target_data)

            x = np.arange(-10, 10, 0.001)    
            p = norm.pdf(x, mean_1, std_1)
            q = norm.pdf(x, mean_2, std_2)

            kl_val = KL(p, q)
            kl_result.append(kl_val)
        return kl_result
    
    def get_reference_result(self, source_dataset, target_dataset):
        # get reference result
        mean_1, std_1 = np.mean(source_dataset), np.std(source_dataset)
        mean_2, std_2 = np.mean(target_dataset), np.std(target_dataset)

        x = np.arange(-10, 10, 0.001)    
        p = norm.pdf(x, mean_1, std_1)
        q = norm.pdf(x, mean_2, std_2)
        ref = KL(p, q)
        return  ref
    
    def get_valid_instance_index(self, source_dataset, target_dataset):
        
        ref = self.get_reference_result(source_dataset, target_dataset)
        kl_result = self.get_kl_result_removing_instances(source_dataset, target_dataset)
        filtered_idx = np.array(kl_result) > ref

        return filtered_idx
