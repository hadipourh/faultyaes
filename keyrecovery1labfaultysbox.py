"""
Copyright (C) 2021  Hosein Hadipour
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Key recovery 1, for given faulty S-boxes from the laboratory.
"""

from faultyaes import *
import numpy as np
from statistics import mean, variance
import random
import itertools
import time

true_sb = [
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
        ]
## The following faulty sboxes have been derived in the laboratory
sb_look_up_2bytes_faults_str = "637c777bf26b6fc53001672bfed7abf6ca82c97dfa5947f6add4a2af9ca472c0b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b27509832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cfd0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdbe0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9ee1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16"
sb_look_up_4bytes_faults_str = "637c777bf26b6fc53001672bfed7ab7eca82c97dfa79f7f7add4a2af9ca472c0b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b27509832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cfd0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdbe0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9ee1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16"
sb_look_up_6bytes_faults_str = "637c777bf26b6fc53001672bfedfffffca82c97dfaffffffadd4a2af9ca472c0b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b27509832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cfd0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdbe0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9ee1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16"
sb_look_up_8bytes_faults_str = "637c777bf26b6fc5300167abfedfbf76ca82c9fdfffdfff7add4a2af9ca472c0b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b27509832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cfd0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdbe0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9ee1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16"
sb_look_up_2bytes_faults = [int(sb_look_up_2bytes_faults_str[2*i:2*i+2], 16) for i in range(256)]
sb_look_up_4bytes_faults = [int(sb_look_up_4bytes_faults_str[2*i:2*i+2], 16) for i in range(256)]
sb_look_up_6bytes_faults = [int(sb_look_up_6bytes_faults_str[2*i:2*i+2], 16) for i in range(256)]
sb_look_up_8bytes_faults = [int(sb_look_up_8bytes_faults_str[2*i:2*i+2], 16) for i in range(256)]

def generate_data(number_of_faults=2, bias=800):
    """
    Generate candidates for delta
    """

    m = 2**8 - number_of_faults
    expected_number_of_queries = int(np.ceil((m*harmonic_number(m))))
    number_of_random_plaintexts = expected_number_of_queries + bias
    produced_ciphertexts = []
    reference_set = set(list(range(256)))
    ##################################################################
    # Initialize a faulty AES for this experiment
    observed_bytes = [[[] for _ in range(4)] for _ in range(4)]
    non_observed_bytes = [[[] for _ in range(4)] for _ in range(4)]
    master_key = random.getrandbits(128)
    faulty_aes = AES(master_key)
    last_round_key = faulty_aes.round_keys[4*10:4*11]
    last_round_key = [last_round_key[j][i] for j in range(4) for i in range(4)]
    if number_of_faults == 2:
        faulty_aes.apply_fault_lab(sb_look_up_2bytes_faults)
    elif number_of_faults == 4:
        faulty_aes.apply_fault_lab(sb_look_up_4bytes_faults)
    elif number_of_faults == 6:
        faulty_aes.apply_fault_lab(sb_look_up_6bytes_faults)
    elif number_of_faults == 8 or number_of_faults == 7:
        faulty_aes.apply_fault_lab(sb_look_up_8bytes_faults)
    else:
        return "Number of faults is assumed to be in [2, 4, 6, 8] in this simulation!"
    known_ciphertexts = []
    for this_query in range(number_of_random_plaintexts):
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
    #print("Expected number of queries: %d, bias: %d" % (expect_number_of_queries, bias))
    D = [[] for _ in range(16)]
    for col in range(4):
        for row in range(4):
            j = 4*col + row
            D[j] = non_observed_bytes[col][row]
    return known_ciphertexts, D, faulty_aes.dictionary_of_replacement, last_round_key

def find_delta_candidates(D0, Dj, number_of_faults):
    """
    Implement Algorithm 2: find deltaj = skR0 + skRj for limited number of given ciphertexts
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

def generate_candidates_k_v(nf, bias):
    known_ciphertexts, D, fault_mapping, last_round_key = generate_data(number_of_faults=nf, bias=bias)
    delta_candidates = []
    for position in range(16):
        deltaj = find_delta_candidates(D[0], D[position], number_of_faults=nf)
        delta_candidates.append(deltaj)
    all_possible_delta_vectors = list(itertools.product(*delta_candidates))
    k_v_candidates = dict()
    for sk0 in range(0, 256):
        for delta_vector in all_possible_delta_vectors:
            # print("Delta vector: %s" % [delta for delta in delta_vector])
            k_v_candidates[tuple([sk0 ^ delta for delta in delta_vector])] = [sk0 ^ d for d in D[0]]
    return known_ciphertexts, k_v_candidates, last_round_key, fault_mapping

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
    if number_of_faults == 2:
        faulty_aes.apply_fault_lab(sb_look_up_2bytes_faults)
    elif number_of_faults == 4:
        faulty_aes.apply_fault_lab(sb_look_up_4bytes_faults)
    elif number_of_faults == 6:
        faulty_aes.apply_fault_lab(sb_look_up_6bytes_faults)
    elif number_of_faults == 8 or number_of_faults == 7:
        faulty_aes.apply_fault_lab(sb_look_up_8bytes_faults)
    else:
        return "Number of faults is assumed to be in [2, 4, 6, 8] in this simulation!"
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

def compute_avg_cnt_for_wrong_and_correct_keys(number_of_faults=4, number_of_independent_experiments=100):
    m = 256 - number_of_faults
    number_of_known_ciphertexts = int(np.ceil(m*harmonic_number(m)))
    number_of_derived_keys = []
    cnt_of_correct_keys = []
    all_cnt_of_wrong_keys = []
    true_and_retrievd_last_round_keys = dict()
    for nxp in range(number_of_independent_experiments):
        D = [[]]
        while len(D[0]) != number_of_faults:
            known_ciphertexts, k_v_candidates, last_round_key, fault_mapping, D \
                = generate_input_data_for_key_recovery(number_of_faults, number_of_known_ciphertexts)
        counter_Ki_Vi = dict()
        aes_instance = AES(0)
        if number_of_faults == 2:
            aes_instance.apply_fault_lab(sb_look_up_2bytes_faults)
        elif number_of_faults == 4:
            aes_instance.apply_fault_lab(sb_look_up_4bytes_faults)
        elif number_of_faults == 6:
            aes_instance.apply_fault_lab(sb_look_up_6bytes_faults)
        elif number_of_faults == 8 or number_of_faults == 7:
            aes_instance.apply_fault_lab(sb_look_up_8bytes_faults)
        else:
            return "Number of faults is assumed to be in [2, 4, 6, 8] in this simulation!"
        number_of_candidates = len(k_v_candidates.keys())
        print("Number of faults: %d, Number of known ciphertexts: %d, Number of key candidates: %d" %\
             (number_of_faults, len(known_ciphertexts), number_of_candidates))
        print("----------------- START KEY RECOVERY -----------------")
        progress_bar = 0
        start_time = time.time()        
        for Ki in k_v_candidates.keys():
            if progress_bar % 50 == 0:
                print('Number of faults: %2d, Candidate No: %7d / %7d - Experiment No: %3d / %3d' %\
                     (number_of_faults, progress_bar, number_of_candidates, (nxp+1), number_of_independent_experiments))
            counter_Ki_Vi[Ki] = 0
            Ki_matrix = [[Ki[i + 4*j] for i in range(4)] for j in range(4)]
            aes_instance.derive_round_keys_from_last_round_key(Ki_matrix)
            for this_cipher in known_ciphertexts:
                counter_Ki_Vi[Ki] += aes_instance.decrypt_and_count1(this_cipher, k_v_candidates[Ki])
            progress_bar += 1
        max_cnt = max(counter_Ki_Vi.values())
        derived_keys = [K for K in counter_Ki_Vi.keys() if counter_Ki_Vi[K] == max_cnt]
        elapsed_time = time.time() - start_time
        print("Time used by key recovery: %0.2f Seconds" % elapsed_time)
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

if __name__ == "__main__":
    true_and_retrievd_last_round_keys, output_dict =\
    compute_avg_cnt_for_wrong_and_correct_keys(number_of_faults=6, 
                                               number_of_independent_experiments=1)
    output = output_dict["avg_number_of_derived_keys"]
    print(f"Average number of derived keys: {output}")
    output = output_dict["avg_cnt_of_correct_keys"]
    print(f"Average value for the counter of correct keys: {output}")
    output = output_dict["avg_cnt_of_wrong_keys"]
    print(f"Average value for the counter of wrong keys: {output}")