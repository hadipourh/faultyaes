"""
In this experiment we aim to check how Algorithm 1 
in our paper works in practice
"""

from faultyaes import *
from experiment1 import harmonic_number
import matplotlib.pyplot as plt
import numpy as np

def alg1_to_find_delta(D0, Dj, number_of_faults):
    """
    Implement algorithm 1 of our paper

    :param int[] D0: list of non-observed values in output byte 0
    :param int[] Dj: list of non-observed values in output byte j
    :return: candidates for deltaj = skR[0] + skR[j] as well as a bar plot
    :rtype: list of integers as as well as pyplot object
    """

    delta_counters = dict()
    for ell in range(number_of_faults):
        alpha_l = D0[0] ^ Dj[ell]
        delta_counters[alpha_l] = 1
        Dtemp = set(Dj).difference(set([Dj[ell]]))
        for i in range(1, number_of_faults):
            E = D0[i] ^ alpha_l
            if E in Dtemp:
                delta_counters[alpha_l] += 1
                Dtemp = Dtemp.difference(set([E]))
    keys, values = delta_counters.keys(), delta_counters.values()
    plt.bar(keys, values, width=2)    
    candidates = [delta for delta in delta_counters.keys() if delta_counters[delta] == number_of_faults]
    plt.bar(candidates, [1]*len(candidates))
    return candidates, plt

def generate_deltas_for_large_number_of_ciphertexts(number_of_faults=2):
    """
    Run faulty aes for sufficiently large number of 
    random plaintexts and collect the observed as well as 
    non-observed values for each output byte

    :param int number_of_faults: number of faults
    :return: number of queries (number_of_random_plaintexts)
             list of non-observed values at each position (D)
             fault mapping (faulty_aes.dictionary_of_replacement)
             last round key corresponding to the chosen random key (last_round_key)
    """

    m = 2**8 - number_of_faults
    number_of_random_plaintexts = 2*int(np.ceil(m*harmonic_number(m)))
    reference_set = set(list(range(256)))
    ##################################################################
    # Initialize a faulty AES for this experiment
    observed_bytes = [[[] for _ in range(4)] for _ in range(4)]
    non_observed_bytes = [[[] for _ in range(4)] for _ in range(4)]
    master_key = random.getrandbits(128)
    faulty_aes = AES(master_key)
    last_round_key = faulty_aes.round_keys[4*10:4*11]
    last_round_key = [last_round_key[j][i] for j in range(4) for i in range(4)]
    faulty_aes.apply_fault(number_of_faults)
    for this_query in range(number_of_random_plaintexts):
        # Choose a plaintext at random
        plaintext = random.getrandbits(128)
        ciphertext = faulty_aes.encrypt(plaintext)
        ciphertext = text2matrix(ciphertext)
        for col in range(4):
            for row in range(4):
                observed_bytes[col][row].append(ciphertext[col][row])
    for col in range(4):
        for row in range(4):
            observed = set(observed_bytes[col][row])
            non_observed_bytes[col][row] = list(reference_set.difference(observed))
    ##################################################################
    D = [[] for _ in range(16)]
    for col in range(4):
        for row in range(4):
            j = 4*col + row
            D[j] = non_observed_bytes[col][row]
            # print('D_{%d}: %s' % (j, D[j]))
    return number_of_random_plaintexts, D, faulty_aes.dictionary_of_replacement, last_round_key

if __name__ == "__main__":
    nf = 4
    position = 5
    number_of_ciphertexts, DD, true_fault_mapping, true_last_round_key =\
        generate_deltas_for_large_number_of_ciphertexts(number_of_faults=nf)
    delta_candidates, diagram = alg1_to_find_delta(DD[0], DD[position], number_of_faults=nf)
    print("Number of available ciphertexts: %d" % number_of_ciphertexts)
    print("Candidates for delta%d: %s" % (position, delta_candidates))
    print("skR[0] xor skR[%d]: %d" % (position, true_last_round_key[0] ^ true_last_round_key[position]))
    diagram.show()
