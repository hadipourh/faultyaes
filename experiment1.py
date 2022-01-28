"""
Assuming that number_of_faults faults have been applied, we generate 
N random (faulty) ciphertexts and collect the observed values at each 
output byte of ciphertext. 
We repeat this experiment for several random master keys to see how 
many random queries is required on average to observe all possible 
values at least once for an arbitrary output byte of ciphertext.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from faultyaes import *



def experiment1(number_of_experiments, number_of_faults, number_of_queries_in_each_exper):
    """
    Assuming that number_of_faults faults have been applied, we generate 
    N random (faulty) ciphertexts and collect the observed values at each 
    output byte of ciphertext. 
    We repeat this experiment for several random master keys to see how 
    many random queries is required on average to observe all possible 
    values at least once for an arbitrary output byte of ciphertext.

    :param int number_of_experiments: number of random independent experiments
    :param int number_of_faults: number of faults in each random experiment
    :param int number_of_queries_in_each_exper: number of queries in each single random experiment
    :return: the average of non-observed values at each output byte
             as well as the total average of non-observed values over the 16 bytes
    :rtype: float, float
    """

    produced_ciphertexts = []
    number_of_observed_bytes_in_this_experiment = [[[] for _ in range(4)] for _ in range(4)]
    number_of_non_observed_bytes_in_this_experiment = [[[] for _ in range(4)] for _ in range(4)]
    reference_set = set(list(range(256)))
    for this_experiment in range(number_of_experiments):
        # Initialize a faulty AES for this experiment
        observed_bytes = [[[] for _ in range(4)] for _ in range(4)]
        master_key = random.getrandbits(128)
        faulty_aes = AES(master_key)
        faulty_aes.apply_fault(number_of_faults)
        for this_query in range(number_of_queries_in_each_exper):
            # Choose a plaintext at random
            plaintext = random.getrandbits(128)
            ciphertext = faulty_aes.encrypt(plaintext)
            ciphertext = text2matrix(ciphertext)
            for row in range(4):
                for col in range(4):
                    observed_bytes[col][row].append(ciphertext[col][row])
        for row in range(4):
            for col in range(4):
                observed = set(observed_bytes[col][row])
                non_observed = reference_set.difference(observed)
                # number_of_observed_bytes_in_this_experiment[row][col].append(len(observed))
                number_of_non_observed_bytes_in_this_experiment[col][row].append(len(non_observed))
        # Generate the set of differences for the the first cell
    mean_output = [[0 for _ in range(4)] for _ in range(4)]
    total_mean = 0
    for row in range(4):
        for col in range(4):
            mean_for_each_position = np.mean(number_of_non_observed_bytes_in_this_experiment[row][col])
            mean_output[col][row] = mean_for_each_position
            total_mean += mean_for_each_position
    total_mean = total_mean / 16.0
    return mean_output, total_mean

if __name__ == "__main__":
    """
    Perform experiment 1 to estimate how many queries is sufficient to 
    see all of the possible values at least once in each ciphertext's byte
    """
    number_of_faults = int(sys.argv[1]) if (len(sys.argv) > 1) else 5    
    bias = 0
    m = 2**8 - number_of_faults
    expected_number_of_queries = int(np.ceil((m*harmonic_number(m)))) + bias
    mean_output, total_mean = experiment1(number_of_experiments=10,\
                                        number_of_faults=number_of_faults,\
                                        number_of_queries_in_each_exper=expected_number_of_queries)
    output = "Number of non-observed values on average:"
    for row in range(4):
        output += "\n"
        for col in range(4):
            output += "%0.02f, " % mean_output[row][col]
    plt.imshow(mean_output)
    plt.xticks(range(4), range(4))
    plt.yticks(range(4), range(4))
    plt.colorbar()    
    print(output)
    print(f"Number of faults: {number_of_faults}")
    print(f"Number of queries in each experiment: {expected_number_of_queries}")
    plt.show()