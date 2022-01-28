# Multiple Persisten Faults Attack - Key Recovery Based on V and V*

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
import random
import itertools
import time
```

## Generate Candidates for delta

### Find deltaj = skR0 + skRj Assuming that Enough Number of Known Ciphertexts Are Available


```python
def find_delta_candidates(D0, Dj, number_of_faults):
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
    candidates = [delta for delta in delta_counters.keys() if delta_counters[delta] == number_of_faults]
    return candidates
```


```python
def find_D0_star(number_of_faults, D, D_star_candidates, last_round_key):
    D0_star = D_star_candidates[0]
    counter_of_values = dict()
    for x in range(256):
        counter_of_values[x] = 0
    for j in range(1, 16):
        deltaj = find_delta_candidates(D[0], D[j], number_of_faults=number_of_faults)
        if len(deltaj) != 1:
            print("Size of deltaj%d = %d!"% (j, len(deltaj)))
            return
        deltaj = deltaj[0]
        Dpj = [d ^ deltaj for d in D_star_candidates[j]]
        for x in Dpj:
            counter_of_values[x] += 1
        D0_star = [d for d in D0_star if d in  Dpj]        
    output_temp = {k: v for k, v in \
                sorted(counter_of_values.items(), key=lambda item: item[1], reverse=True)}
    output = list(output_temp.keys())[0:number_of_faults]
    for candidate in output:
        print("candidate: %d, counter: %d" % (candidate, output_temp[candidate]))
    return output
```


```python
def generate_input_data_for_key_recovery(number_of_faults, number_of_known_ciphertexts):
    ##################################################################
    # Initialize a faulty AES for this experiment
    byte_observation_counter = [[dict() for _ in range(4)] for _ in range(4)]
    for col in range(4)  :
        for row in range(4):
            for x in range(256):
                byte_observation_counter[col][row][x] = 0
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
                byte_observation_counter[col][row][ciphertext[col][row]] += 1
    ##################################################################
    D = [0 for _ in range(16)]
    D_star = [0 for _ in range(16)]
    for col in range(4):
        for row in range(4):
            j = 4*col + row
            temp = {k: v for k, v in \
                sorted(byte_observation_counter[col][row].items(), key=lambda item: item[1], reverse=True)}
            temp = list(temp.keys())
            #######################################XXXXXXXXXXXXXXXXXXXX###############################
            D_star[j] = temp[0:2*number_of_faults]            
            D[j] = [x for x in range(256) if byte_observation_counter[col][row][x] == 0]
    delta_candidates = []
    for position in range(16):
        deltaj = find_delta_candidates(D[0], D[position], number_of_faults=number_of_faults)
        delta_candidates.append(deltaj)
    all_possible_delta_vectors = list(itertools.product(*delta_candidates))
    k_v_candidates = dict()
    print("Number of delta candidates: %d" % len(all_possible_delta_vectors))
    for sk0 in range(0, 256):
        for delta_vector in all_possible_delta_vectors:
            k_v_candidates[tuple([sk0 ^ delta for delta in delta_vector])] = \
                             [[sk0 ^ d for d in D[0]], [sk0 ^ last_round_key[0] ^ d for d in fault_mapping.values()], 1]
    return known_ciphertexts, k_v_candidates, last_round_key, fault_mapping, D, D_star
```

### Check the Quality of Our Algorithm for Dtecting D0*


```python
nf = 4
m = 2**8 - nf
number_of_ciphertexts = 2*int(np.ceil(m*harmonic_number(m)))
ciphertexts, k_candidates, true_last_round_key, fault_mapping, DD, DD_star =\
     generate_input_data_for_key_recovery(number_of_faults=nf, number_of_known_ciphertexts=number_of_ciphertexts)
D0_star = find_D0_star(number_of_faults=nf, D=DD, D_star_candidates=DD_star, last_round_key=true_last_round_key)
print("D0_star: %s" % D0_star)
print("True fault mapping: %s" % fault_mapping)
```

    Number of delta candidates: 1
    candidate: 8, counter: 14
    candidate: 124, counter: 14
    candidate: 159, counter: 14
    candidate: 2, counter: 11
    D0_star: [8, 124, 159, 2]
    True fault mapping: {77: 126, 252: 116, 237: 0, 6: 227}


### Provide Required Data for Key recovery


```python
def compute_avg_cnt_for_wrong_and_correct_keys(number_of_faults=4, number_of_independent_experiments=100):
    m = 256 - number_of_faults
    number_of_known_ciphertexts = 2*int(np.ceil(m*harmonic_number(m)))
    number_of_derived_keys = []
    cnt_of_correct_keys = []
    all_cnt_of_wrong_keys = []
    output_dict = dict()
    true_and_retrievd_last_round_keys = dict()
    for nxp in range(number_of_independent_experiments):
        D = [[]]
        known_ciphertexts, k_v_candidates, last_round_key, fault_mapping, D, D_star\
                 = generate_input_data_for_key_recovery(number_of_faults=number_of_faults, number_of_known_ciphertexts=number_of_known_ciphertexts)
        while len(D[0]) != number_of_faults or len(k_v_candidates) == 0:
            known_ciphertexts, k_v_candidates, last_round_key, fault_mapping, D, D_star\
                 = generate_input_data_for_key_recovery(number_of_faults=number_of_faults, number_of_known_ciphertexts=number_of_known_ciphertexts)
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
                     (number_of_faults, progress_bar, number_of_candidates, nxp, number_of_independent_experiments))
            Ki_matrix = [[Ki[i + 4*j] for i in range(4)] for j in range(4)]
            aes_instance.derive_round_keys_from_last_round_key(Ki_matrix)
            for cipher_count in range(256):
                this_cipher = known_ciphertexts[cipher_count]
                k_v_candidates[Ki][2] = aes_instance.decrypt_and_count2(this_cipher, k_v_candidates[Ki][0], k_v_candidates[Ki][1])
                if k_v_candidates[Ki][2] == 0:
                    break
            progress_bar += 1
        derived_keys = [K for K in k_v_candidates.keys() if k_v_candidates[K][2] == 1]
        print("size D: %d" % len(D[0]))
        elapsed_time = time.time() - start_time
        print("Time used by key recovery: %0.2f Seconds" % elapsed_time)
        print("------------- KEY RECOVERY WAS FINISHED -------------")
        number_of_derived_keys.append(len(derived_keys))
        true_and_retrievd_last_round_keys[tuple(last_round_key)] = derived_keys
    return true_and_retrievd_last_round_keys, number_of_derived_keys
```

### Key Recovery


```python
true_and_retrievd_last_round_keys, number_of_derived_keys =\
     compute_avg_cnt_for_wrong_and_correct_keys(number_of_faults=4, number_of_independent_experiments=1)
print(number_of_derived_keys)
print(true_and_retrievd_last_round_keys)
```

    Number of delta candidates: 1
    Number of faults: 4, Number of known ciphertexts: 3080, Number of key candidates: 256
    ----------------- START KEY RECOVERY -----------------
    Number of faults:  4, Candidate No:       0 /     256 - Experiment No:   0 /   1
    Number of faults:  4, Candidate No:      50 /     256 - Experiment No:   0 /   1
    Number of faults:  4, Candidate No:     100 /     256 - Experiment No:   0 /   1
    Number of faults:  4, Candidate No:     150 /     256 - Experiment No:   0 /   1
    Number of faults:  4, Candidate No:     200 /     256 - Experiment No:   0 /   1
    Number of faults:  4, Candidate No:     250 /     256 - Experiment No:   0 /   1
    size D: 4
    Time used by key recovery: 0.09 Seconds
    ------------- KEY RECOVERY WAS FINISHED -------------
    [1]
    {(239, 107, 167, 155, 36, 167, 209, 205, 88, 227, 243, 23, 205, 46, 114, 251): [(239, 107, 167, 155, 36, 167, 209, 205, 88, 227, 243, 23, 205, 46, 114, 251)]}

