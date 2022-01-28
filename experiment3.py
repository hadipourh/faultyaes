"""
In this experiment we aim to implement the algorithm 2 
to see how it works in practice.
"""

from experiment1 import harmonic_number
import random
import numpy as np
from faultyaes import *
from math import exp, log2
from statistics import mean

def find_delta_candidates(D0, Dj, number_of_faults):
    """
    Implement algorithm 2 to find to find deltaj = skr0 + skrj when 
    a limited number of known ciphertexts is available

    :param int[] D0: list of non-observed values in output byte 0
    :param int[] Dj: list of non-observed values in output byte j
    :return: candidates for deltaj = skR[0] + skR[j]
    :rtype: list of integers
    """


    lambda_prime = len(Dj)
    lambda_prime_zero = len(D0)
    final_candidates = []
    for k in range(lambda_prime_zero - number_of_faults + 1): # Iterating up to this number ensures a non-empty output
        candidates = []
        delta_counters = dict()
        for ell in range(lambda_prime):
            alpha_l = D0[k] ^ Dj[ell]
            delta_counters[alpha_l] = 1
            Dtemp = set(Dj).difference(set([Dj[ell]]))
            D0_complement = [d for d in D0 if d != D0[k]]
            for d in D0_complement:
                E = d ^ alpha_l
                if E in Dtemp:
                    delta_counters[alpha_l] += 1                    
                    Dtemp = Dtemp.difference(set([E]))
        candidates = [delta for delta in delta_counters.keys() if delta_counters[delta] >= number_of_faults]
        final_candidates.extend(candidates)
        final_candidates = list(set(final_candidates))
    return final_candidates


def generate_deltas(number_of_faults=2, number_of_non_observed_values=3):
    """
    Given the number of faults, simulate the situation where 
    the number of nonobserved values is equal to a certain value

    :param int number_of_faults: number of faults
    :param int number_of_non_observed_values: number of nonobserved values
    :return: number of queries (number_of_random_plaintexts)
             list of non-observed values at each position (D)
             fault mapping (faulty_aes.dictionary_of_replacement)
             last round key corresponding to the chosen random key (last_round_key)
    """

    m = 2**8 - number_of_faults
    m_p = 2**8 - number_of_non_observed_values
    number_of_random_plaintexts = int(np.ceil((m*(harmonic_number(m) - harmonic_number(m - m_p)))))
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
    return number_of_random_plaintexts, D, faulty_aes.dictionary_of_replacement, last_round_key

def compute_lamda_prime_from_lambda_and_N(N, lam):
    """
    Given the number of available ciphertexts, approximate the number 
    of nonobserved values

    :param int N: number of available ciphertexts
    :param int lam: number of faults
    :return: estimation of nonobserved values
    :rtype: float
    """

    a = 2**8 - lam
    b = 1.0/(2**8 - lam)
    c = lam
    output = a*exp(-b*N) + c
    return output

if __name__ == "__main__":
    lam = 2
    lam_p = 4
    number_of_trails = 100
    number_of_candidates = []
    for i in range(number_of_trails):
        number_of_known_ciphertexts, D, fault_dictionary, last_round_key = generate_deltas(number_of_faults=lam, number_of_non_observed_values=lam_p)
        lam_p_positions = [k for k in range(16) if len(D[k]) == lam_p and k != 0]
        while lam_p_positions == [] or len(D[0]) != lam_p:
            number_of_known_ciphertexts, D, fault_dictionary, last_round_key = generate_deltas(number_of_faults=lam, number_of_non_observed_values=lam_p)
            lam_p_positions = [k for k in range(16) if len(D[k]) == lam_p and k != 0]
        for position in lam_p_positions:
            output = find_delta_candidates(D[0], D[position], number_of_faults=lam)
            number_of_candidates.append(len(output))
        print("Experiment No: %3d" % (i + 1))
    print(f"Number of known ciphertexts: {number_of_known_ciphertexts}")
    print("Average number of candidates for deltaj in each output byte: %0.2f" % mean(number_of_candidates))    
    mean_num_of_candidates = mean(number_of_candidates)
    a = log2((mean_num_of_candidates**15)*256)
    print("Number of key candidates: 2^(%0.02f)" % a)