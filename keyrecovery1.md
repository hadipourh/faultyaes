# Multiple Persisten Faults Attack - Key Recovery

## License

```
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
```

## Include Faulty AES and Other Required Tools


```python
%load_ext autoreload
%autoreload 2
from faultyaes import *
import numpy as np
from statistics import mean, variance
import random
import itertools
import time
```

### Check to see whether the `derive_round_keys_from_last_round_key` function works correctly


```python
master_key = 0x2B7E151628AED2A6ABF7158809CF4F3C
plaintext = 0x6BC1BEE22E409F96E93D7E117393172A
expected_ciphertext = 0x3AD77BB40D7A3660A89ECAF32466EF97
faulty_aes = AES(master_key)
faulty_aes.apply_fault(number_of_faults=0)
last_round_key = faulty_aes.round_keys[4*10:4*11]
faulty_aes.derive_round_keys_from_last_round_key(last_round_key)
ciphertext = faulty_aes.encrypt(plaintext)
if ciphertext == expected_ciphertext:
    print("derive_round_keys_from_last_round_key` works correctly :-)")
else:
    print("derive_round_keys_from_last_round_key` doesn't work correctly :-)")
```

    derive_round_keys_from_last_round_key` works correctly :-)


### Run `faulty_aes.decrypt_and_count` for test


```python
cnt = faulty_aes.decrypt_and_count1(ciphertext=ciphertext, Vi=[])
print("cnt = %d" % cnt)
```

    cnt = 9


## Experiment 1
In this experiment we aim to implement the key recovery algorithm (algorithm 3) to see how it works in reality

### Generate Candidates for delta


```python
def generate_data(number_of_faults=2, bias=800):
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
    faulty_aes.apply_fault(number_of_faults)
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
```

### Implement Algorithm 2: Find deltaj = skR0 + skRj For Limited Number of Given Ciphertexts


```python
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
```

### Run `generate_data` for test


```python
nf = 4
position = 3
bias = -500
known_ciphertexts, D, fault_mapping, last_round_key = generate_data(number_of_faults=nf, bias=bias)
output = find_delta_candidates(D[0], D[position], number_of_faults=nf)
print("Computed delta%d: %s" % (position, output))
print("skR[0] xor skR[%d]: %d" % (position, last_round_key[0] ^ last_round_key[position]))
```

    Computed delta3: [56]
    skR[0] xor skR[3]: 56


### Collect Candidates for (K, V)
In this experiment we guess the first byte of last round key and determine the remaining key bytes based on the derived candidates for deltaj (where 1 <= j <= 15). 


Let `D[0] = {d_0, d_1, d_2, ..., d_lambda0}`, then for each key candidate Ki we derive the corresponding set of impossible values according to the following relations:

```
V = {d_0 + Ki[0], d_1 + Ki[0], ..., d_lambda0 + Ki[0]}
```
Note that it is oly the first byte of Ki, and the set `D[0]` that are used to derive the corresponding set of impossible values, i.e., Vi.
In summary, for each key guess, we have a corresponding set of impossible values which is denoted by Vi. 


```python
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
```

## Key Recovery


```python
# bias = 100
# nf = 4
# start_time = time.time()
# print("Running a faulty AES with %d faults to generate some random ciphertexts ..." % nf)
# known_ciphertexts, k_v_candidates, last_round_key, fault_mapping = generate_candidates_k_v(nf, bias)
# elapsed_time = time.time() - start_time
# print("Known ciphertexts were generated after %0.2f seconds." % elapsed_time)
# print("Number of known ciphertexts: %d" % len(known_ciphertexts))
# print("Number of (K, V) candidates: %d" % len(k_v_candidates.keys()))
# counter_Ki_Vi = dict()
# counters = [0]*len(list(k_v_candidates.keys()))
# aes_instance = AES(0)
# aes_instance.apply_fault(number_of_faults=nf, fault_mapping=fault_mapping)
# start_time = time.time()
# print("----------------- START KEY RECOVERY -----------------")
# for Ki in k_v_candidates.keys():
#     counter_Ki_Vi[Ki] = 0
#     Ki_matrix = [[Ki[i + 4*j] for i in range(4)] for j in range(4)]
#     aes_instance.derive_round_keys_from_last_round_key(Ki_matrix)
#     for this_cipher in known_ciphertexts:
#         counter_Ki_Vi[Ki] += aes_instance.decrypt_and_count(this_cipher, k_v_candidates[Ki])
# max_cnt = max(counter_Ki_Vi.values())
# derived_keys = [K for K in counter_Ki_Vi.keys() if counter_Ki_Vi[K] == max_cnt]
# elapsed_time = time.time() - start_time
# print("Time used by key recovery: %0.2f Seconds" % elapsed_time)
# print("------------- KEY RECOVERY WAS FINISHED -------------")
# print("Derived (last-round) key(s):")
# for K in derived_keys:
#     print("%s,\t counter: %d" % (K, counter_Ki_Vi[K]))
# print("True (last-round) key:\n%s" % last_round_key)
```


```python
# print("Correct key's counter: %d" % max_cnt)
# counters = list(counter_Ki_Vi.values())
# counters.remove(max_cnt)
# mean_of_wrong_counters = np.mean(counters)
# print("Mean of wrong keys' counter: %0.2f" % mean_of_wrong_counters)
```

## Computing Average Values for Key Recovery Parameters


```python
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
```


```python
# known_ciphertexts, k_v_candidates, last_round_key, fault_mapping, D \
#                 = generate_input_data_for_key_recovery(3, 1500)
# print(D)
```


```python
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
        aes_instance.apply_fault(number_of_faults=number_of_faults, fault_mapping=fault_mapping)
        number_of_candidates = len(k_v_candidates.keys())
        print("Number of faults: %d, Number of known ciphertexts: %d, Number of key candidates: %d" %\
             (number_of_faults, len(known_ciphertexts), number_of_candidates))
        print("----------------- START KEY RECOVERY -----------------")
        progress_bar = 0
        start_time = time.time()        
        for Ki in k_v_candidates.keys():
            if progress_bar % 50 == 0:
                print('Number of faults: %2d, Candidate No: %7d / %7d - Experiment No: %3d / %3d' %\
                     (number_of_faults, progress_bar, number_of_candidates, (nxp + 1), number_of_independent_experiments))
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
    output_dict["variance_cnt_of_correct_keys"] = variance(cnt_of_correct_keys)
    output_dict["variance_cnt_of_wrong_keys"] = variance(all_cnt_of_wrong_keys)
    return true_and_retrievd_last_round_keys, output_dict
    
```


```python
true_and_retrievd_last_round_keys, output_dict =\
     compute_avg_cnt_for_wrong_and_correct_keys(number_of_faults=5, number_of_independent_experiments=2)
```

    Number of faults: 5, Number of known ciphertexts: 1533, Number of key candidates: 256
    ----------------- START KEY RECOVERY -----------------
    Number of faults:  5, Candidate No:       0 /     256 - Experiment No:   1 /   2
    Number of faults:  5, Candidate No:      50 /     256 - Experiment No:   1 /   2
    Number of faults:  5, Candidate No:     100 /     256 - Experiment No:   1 /   2
    Number of faults:  5, Candidate No:     150 /     256 - Experiment No:   1 /   2
    Number of faults:  5, Candidate No:     200 /     256 - Experiment No:   1 /   2
    Number of faults:  5, Candidate No:     250 /     256 - Experiment No:   1 /   2
    Time used by key recovery: 20.05 Seconds
    ------------- KEY RECOVERY WAS FINISHED -------------
    Number of faults: 5, Number of known ciphertexts: 1533, Number of key candidates: 256
    ----------------- START KEY RECOVERY -----------------
    Number of faults:  5, Candidate No:       0 /     256 - Experiment No:   2 /   2
    Number of faults:  5, Candidate No:      50 /     256 - Experiment No:   2 /   2
    Number of faults:  5, Candidate No:     100 /     256 - Experiment No:   2 /   2
    Number of faults:  5, Candidate No:     150 /     256 - Experiment No:   2 /   2
    Number of faults:  5, Candidate No:     200 /     256 - Experiment No:   2 /   2
    Number of faults:  5, Candidate No:     250 /     256 - Experiment No:   2 /   2
    Time used by key recovery: 20.21 Seconds
    ------------- KEY RECOVERY WAS FINISHED -------------



```python
output_dict["avg_number_of_derived_keys"], \
output_dict["avg_cnt_of_correct_keys"], \
output_dict["avg_cnt_of_wrong_keys"], \
output_dict["variance_cnt_of_correct_keys"], \
output_dict["variance_cnt_of_wrong_keys"]
```




    (1, 8069, 3898.5313725490196, 1458, 11546.988208328517)




```python
# if __name__ == '__main__':
#     print('Now in the main code. Process name is:', __name__)
#     number_of_faults_vector = [2]#list(range(3, 17)) + [32, 48, 64]
#     number_of_independent_experiments = 10
#     flag = 'read_data'
#     if flag == 'compute_data':
#         with Pool(len(number_of_faults_vector)) as pool:
#             arguments = [(nf, number_of_independent_experiments) for nf in number_of_faults_vector]
#             results = pool.starmap(compute_avg_cnt_for_wrong_and_correct_keys, arguments)
#         with open('output', 'wb') as f:
#             pickle.dump(results, f)
#     elif flag == 'read_data':
#         with open('output64', 'rb') as f:
#             results = pickle.load(f)
    
```


```python
# ks = list(range(3, 17)) + [32, 48, 64]
# vs = list(range(len(ks)))
# indice = dict(zip(ks, vs))
# n = 64
# results[indice[n]][1]["avg_number_of_derived_keys"], \
# results[indice[n]][1]["avg_cnt_of_correct_keys"], \
# results[indice[n]][1]["avg_cnt_of_wrong_keys"], \
# results[indice[n]][1]["variance_cnt_of_correct_keys"], \
# results[indice[n]][1]["variance_cnt_of_wrong_keys"]
```
