#!/usr/bin/env python
# coding: utf-8

# # Multiple Persisten Faults Attack - Parallel Key Recovery

# ## License
# 
# ```
# Copyright (C) 2021  Hosein Hadipour
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ```

#Required Packages

# In[10]:

from faultyaes import *
import numpy as np
from statistics import mean
import random
import itertools
from fractions import Fraction
import time
from multiprocessing import Pool
import pickle
from os import getpid


# ## Experiment 1
# In this experiment we aim to implement the key recovery algorithm (algorithm 3) to see how it works in practice

# ### Implement Algorithm 2: Find deltaj = skR0 + skRj For Limited Number of Given Ciphertexts

# In[11]:


def find_delta_candidates(D0, Dj, number_of_faults):    
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


# ### Collect Candidates for (K, V)
# In this experiment we guess the first byte of last round key and determine the remaining key bytes based on the derived candidates for deltaj (where 1 <= j <= 15). 
# 
# 
# Let `D[0] = {d_0, d_1, d_2, ..., d_lambda0}`, then for each key candidate Ki we derive the corresponding set of impossible values according to the following relations:
# 
# ```
# V = {d_0 + Ki[0], d_1 + Ki[0], ..., d_lambda0 + Ki[0]}
# ```
# Note that it is oly the first byte of Ki, and the set `D[0]` that are used to derive the corresponding set of impossible values, i.e., Vi.
# In summary, for each key guess, we have a corresponding set of impossible values which is denoted by Vi. 

# ## Key Recovery

# In[12]:


def generate_input_data_for_key_recovery(number_of_faults, number_of_known_ciphertexts):
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
    fault_mapping = faulty_aes.dictionary_of_replacement
    known_ciphertexts = []
    for this_query in range(number_of_known_ciphertexts):
        # Choose a plaintext at random
        plaintext = random.getrandbits(128)
        ciphertext = faulty_aes.encrypt(plaintext)
        known_ciphertexts.append(ciphertext)
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
    delta_candidates = []
    for position in range(16):
        deltaj = find_delta_candidates(D[0], D[position], number_of_faults=number_of_faults)
        delta_candidates.append(deltaj)
    all_possible_delta_vectors = list(itertools.product(*delta_candidates))
    k_v_candidates = dict()
    for sk0 in range(0, 256):
        for delta_vector in all_possible_delta_vectors:
            k_v_candidates[tuple([sk0 ^ delta for delta in delta_vector])] = [sk0 ^ d for d in D[0]]
    return known_ciphertexts, k_v_candidates, last_round_key, fault_mapping, D


# ### Define a Function to Divide the Set of Key Candidates into Some Smaller Sub-stes

# In[13]:


def chunks(data, num_of_chunks=32):
    size_of_each_chunk = len(data) // num_of_chunks
    it = iter(data)
    for i in range(0, len(data), size_of_each_chunk):
        yield {k:data[k] for k in itertools.islice(it, size_of_each_chunk)}


# In[19]:


def check_key_candidates(number_of_faults, fault_mapping, part_of_key_candidates, known_ciphertexts):
    counter_Ki_Vi = dict()
    # pid = current_process().name
    pid = getpid()
    progress_var = 0
    number_of_candidates = len(part_of_key_candidates)
    aes_instance = AES(0)
    aes_instance.apply_fault(number_of_faults=number_of_faults, fault_mapping=fault_mapping)
    for Ki in part_of_key_candidates.keys():
        if progress_var % 50 == 0:
            print(f"process id: {pid}, candidates no {progress_var} / {number_of_candidates}")
        counter_Ki_Vi[Ki] = 0
        Ki_matrix = [[Ki[i + 4*j] for i in range(4)] for j in range(4)]
        aes_instance.derive_round_keys_from_last_round_key(Ki_matrix)
        for this_cipher in known_ciphertexts:
            counter_Ki_Vi[Ki] += aes_instance.decrypt_and_count1(this_cipher, part_of_key_candidates[Ki])
        progress_var += 1
    return counter_Ki_Vi


# In[20]:


def compute_avg_cnt_for_wrong_and_correct_keys(number_of_faults=4, number_of_independent_experiments=10, num_of_processes=16):
    m = 256 - number_of_faults
    number_of_known_ciphertexts = int(np.ceil(m*harmonic_number(m)))
    number_of_derived_keys = []
    cnt_of_correct_keys = []
    all_cnt_of_wrong_keys = []
    true_and_retrievd_last_round_keys = dict()
    for nxp in range(number_of_independent_experiments):
        D = [[]]
        while len(D[0]) != number_of_faults:
            known_ciphertexts, k_v_candidates, last_round_key, fault_mapping, D = generate_input_data_for_key_recovery(number_of_faults, number_of_known_ciphertexts)
        counter_Ki_Vi = dict()
        number_of_candidates = len(k_v_candidates.keys())
        print("Number of faults: %d, Number of known ciphertexts: %d, Number of key candidates: %d" %             (number_of_faults, len(known_ciphertexts), number_of_candidates))

        # Divide the set of key candidates into some smaller chunks
        k_v_candidates_chunks = list(chunks(k_v_candidates, num_of_chunks=num_of_processes))

        print("----------------- START KEY RECOVERY -----------------")
        start_time = time.time()
        # Parallel execution
        with Pool(len(k_v_candidates_chunks)) as pool:
            arguments = [(number_of_faults, fault_mapping, k_v_chunk, known_ciphertexts)                for k_v_chunk in k_v_candidates_chunks]
            results = pool.starmap(check_key_candidates, arguments)
        # End of parallel execution

        # Collect the outputs of parallel processes
        for output in results:
            counter_Ki_Vi.update(output)
        
        max_cnt = max(counter_Ki_Vi.values())
        derived_keys = [K for K in counter_Ki_Vi.keys() if counter_Ki_Vi[K] == max_cnt]
        elapsed_time = time.time() - start_time
        print("Time used by key recovery: %0.2f Seconds, experiment no %2d" % (elapsed_time, nxp))
        print("------------- KEY RECOVERY WAS FINISHED -------------")
        
        number_of_derived_keys.append(len(derived_keys))
        cnt_of_correct_keys.append(max_cnt)
        cnts_of_wrong_keys = [cnt for cnt in counter_Ki_Vi.values() if cnt != max_cnt]
        all_cnt_of_wrong_keys.extend(cnts_of_wrong_keys)
        true_and_retrievd_last_round_keys[derived_keys[0]] = last_round_key
    output_dict = dict()
    output_dict["cnt_of_correct_keys"] = cnt_of_correct_keys
    output_dict["all_cnt_of_wrong_keys"] = all_cnt_of_wrong_keys
    output_dict["avg_number_of_derived_keys"] = mean(number_of_derived_keys)
    output_dict["avg_cnt_of_correct_keys"] = mean(cnt_of_correct_keys)
    output_dict["avg_cnt_of_wrong_keys"] = mean(all_cnt_of_wrong_keys)
    return true_and_retrievd_last_round_keys, output_dict
    


# In[21]:


# true_and_retrievd_last_round_keys, output_dict =\
#      compute_avg_cnt_for_wrong_and_correct_keys(number_of_faults=5, number_of_independent_experiments=2, num_of_processes=4)


# In[22]:


# output_dict["avg_number_of_derived_keys"], \
# output_dict["avg_cnt_of_correct_keys"], \
# output_dict["avg_cnt_of_wrong_keys"], \


# In[24]:


if __name__ == '__main__':
    print('Now in the main code. Process name is:', __name__)
    flag = 'compute_data'
    if flag == 'compute_data':
        results = compute_avg_cnt_for_wrong_and_correct_keys(number_of_faults=3, number_of_independent_experiments=1, num_of_processes=32)
        with open('output_lambda_2', 'wb') as f:
            pickle.dump(results, f)
        print("Number of derived keys: %2d, Counter of correct key: %8d, Counter of wrong key: %8d" % 
        (results[1]["avg_number_of_derived_keys"], \
        results[1]["avg_cnt_of_correct_keys"], \
        results[1]["avg_cnt_of_wrong_keys"]))
    elif flag == 'read_data':
        with open('output_lambda_2', 'rb') as f:
            results = pickle.load(f)

