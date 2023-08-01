import unittest
import numpy as np
 # assuming your function is in a file named my_module.py

class TestComputeTransmittedInfo(unittest.TestCase):
    def test_uniform_distribution(self):
        uniform_mat = np.array([[25, 25], [25, 25]])
        self.assertAlmostEqual(compute_transmitted_info(uniform_mat), 0)

    def test_perfect_clustering(self):
        perfect_mat = np.array([[50, 0], [0, 50]])
        self.assertAlmostEqual(compute_transmitted_info(perfect_mat), 1)

def compute_transmitted_info(conf_mat):
    K = conf_mat.shape[0] # number of classes or clusters
    n = np.sum(conf_mat)  # total number of responses
    nij = conf_mat  # number of responses to stimulus i clustered into stimulus j

    # create an empty array to store the results of each element in the confusion matrix
    h_elements = np.zeros_like(nij, dtype=float)

    # iterate over each element in the confusion matrix
    for i in range(K):
        for j in range(K):
            # skip if there are no responses for this combination
            if nij[i, j] == 0:
                continue

            nkj = np.sum(conf_mat[:, j])  # total number of responses assigned to cluster j
            nik = np.sum(conf_mat[i, :])  # total number of responses to stimulus i
            print(f"For i={i}, j={j}, nkj is of type {type(nkj)} and shape {np.shape(nkj)}")
            print(f"For i={i}, j={j}, nik is of type {type(nik)} and shape {np.shape(nik)}")

            
            # check if nkj or nik are zero
            if nkj == 0:
                term_nkj = 0
            else:
                term_nkj = np.log(nkj)
                
            if nik == 0:
                term_nik = 0
            else:
                term_nik = np.log(nik)
            
            h_elements[i, j] = nij[i, j] * (np.log(nij[i, j]) - term_nkj - term_nik + np.log(n))

    h = (1 / n) * np.sum(h_elements)
    h_normalized = h / np.log(K)  # normalize transmitted information
    return h_normalized
    

if __name__ == "__main__":
    unittest.main()

