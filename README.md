# Multiple Persisten Faults Attack - Faulty AES

This repository complements our research documented in [Practical Multiple Persistent Faults Analysis](https://www.iacr.org/cryptodb/data/paper.php?pubkey=31580) which has been accepted to [IACR-CHES-2022](https://ches.iacr.org/2022/).

ePrint version: [Practical Multiple Persistent Faults Analysis](https://eprint.iacr.org/2021/1362)



![logo](./Figures/close_up_diagram_of_non_observed_values.svg)

- [Multiple Persisten Faults Attack - Faulty AES](#multiple-persisten-faults-attack---faulty-aes)
  - [Implement (Faulty) AES-128](#implement-faulty-aes-128)
  - [Using NIST Test Vectors to Verify the Correctness of AES Implementation](#using-nist-test-vectors-to-verify-the-correctness-of-aes-implementation)
  - [Experiment 1](#experiment-1)
  - [Experiment 2 - Implement Algorithm 1 to Find deltaj = skR[0] + skR[j]](#experiment-2---implement-algorithm-1-to-find-deltaj--skr0--skrj)
  - [Experiment 3 - Implement Algorithm 2 to Find deltaj = skR[0] + skR[j]](#experiment-3---implement-algorithm-2-to-find-deltaj--skr0--skrj)
  - [Algorithm 3 to Find D0*](#algorithm-3-to-find-d0)
  - [Algorithm 4 - Key Recovery 1](#algorithm-4---key-recovery-1)
  - [Parallel Key Recovery 1](#parallel-key-recovery-1)
  - [Algorithm 4 - Key Recovery 1 For Faulty S-boxes Derived in the Laboratory](#algorithm-4---key-recovery-1-for-faulty-s-boxes-derived-in-the-laboratory)
  - [Algorithm 5 - Key Recovery 2](#algorithm-5---key-recovery-2)
  - [Algorithm 5 - Key recovery 2 For Faulty S-boxes Derived in the Laboratory](#algorithm-5---key-recovery-2-for-faulty-s-boxes-derived-in-the-laboratory)
  - [License](#license)

**Required Packages**


```python
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log2
import random
from multiprocessing import Pool
from scipy.optimize import curve_fit
import pickle
from statistics import mean, variance
```

**Initialize Random Generator**

`random.seed(a=None, version=2)`

If a is omitted or None, the current system time is used. If randomness sources are provided by the operating system, they are used instead of the system time (see the os.urandom() function for details on availability).

`random.sample(population, k, *, counts=None)`

Returns a k length list of unique elements chosen from the population sequence or set. Used for random sampling without replacement.

`sample()` is used for random sampling without replacement, and `choices()` is used for random sampling with replacement.

**Generate A Random Faulty S-box**


```python
def gen_random_faulty_sbox(number_of_faults, sbox):
    sbox_values_set = set(sbox)
    random_positions = random.sample(range(0, len(sbox)), number_of_faults)
    original_values = [sbox[i] for i in random_positions]
    original_values_complement = list(sbox_values_set.difference(set(original_values)))
    faulty_values = random.sample(original_values_complement, number_of_faults)
    dictionary_of_replacement = dict(zip(original_values, faulty_values))
    print('Dictionary of replacement: %s' % dictionary_of_replacement)
    faulty_sbox = [0]*len(sbox)
    for i in range(len(sbox)):
        if i in random_positions:
            faulty_sbox[i] = dictionary_of_replacement[sbox[i]]
        else:
            faulty_sbox[i] = sbox[i]
    return faulty_sbox
```

**Define S-box and Generate A Random Faulty S-box**


```python
true_sbox = [
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
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
]
faulty_sbox = gen_random_faulty_sbox(number_of_faults=4, sbox=true_sbox)
print(faulty_sbox)
```

    Dictionary of replacement: {73: 184, 159: 36, 123: 49, 59: 112}
    [99, 124, 119, 49, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 112, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 36, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 184, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]


**Define Harmonic Number**


```python
from fractions import Fraction
harmonic_number = lambda n: float(sum(Fraction(1, d) for d in range(1, n+1)))
```

## Implement (Faulty) AES-128

* In the following implementation it is supposed that key schedule is not effeccted by the faults


```python
xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)

def text2matrix(text):
    matrix = []
    for i in range(16):
        byte = (text >> (8 * (15 - i))) & 0xFF
        if i % 4 == 0:
            matrix.append([byte])
        else:
            matrix[i // 4].append(byte)
    return matrix

def matrix2text(matrix):
    text = 0
    for i in range(4):
        for j in range(4):
            text |= (matrix[i][j] << (120 - 8 * (4 * i + j)))
    return text

class AES:
    def __init__(self, master_key):
        self.Rcon = (
                        0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
                        0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
                        0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
                        0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
                    )
        self.sbox = [
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
        self.change_key(master_key)
    
    def apply_fault(self, number_of_fault):
        self.faulty_sbox = self.gen_random_faulty_sbox(number_of_fault)
        self.inv_faulty_sbox = [0]*(len(self.sbox))
        for i in range(len(self.faulty_sbox)):
            self.inv_faulty_sbox[self.faulty_sbox[i]] = i
    
    def gen_random_faulty_sbox(self, number_of_faults):
        sbox_values_set = set(self.sbox)
        random_positions = random.sample(range(len(self.sbox)), number_of_faults)
        original_values = [self.sbox[i] for i in random_positions]
        original_values_complement = list(sbox_values_set.difference(set(original_values)))
        faulty_values = random.sample(original_values_complement, number_of_faults)
        self.dictionary_of_replacement = dict(zip(original_values, faulty_values))
        faulty_sbox = [0]*256
        for i in range(256):
            if i in random_positions:
                faulty_sbox[i] = self.dictionary_of_replacement[self.sbox[i]]
            else:
                faulty_sbox[i] = self.sbox[i]
        return faulty_sbox

    def change_key(self, master_key):
        self.round_keys = text2matrix(master_key)
        # print(self.round_keys)

        for i in range(4, 4 * 11):
            self.round_keys.append([])
            if i % 4 == 0:
                byte = self.round_keys[i - 4][0]        \
                     ^ self.sbox[self.round_keys[i - 1][1]]  \
                     ^ self.Rcon[i // 4]
                self.round_keys[i].append(byte)

                for j in range(1, 4):
                    byte = self.round_keys[i - 4][j]    \
                         ^ self.sbox[self.round_keys[i - 1][(j + 1) % 4]]
                    self.round_keys[i].append(byte)
            else:
                for j in range(4):
                    byte = self.round_keys[i - 4][j]    \
                         ^ self.round_keys[i - 1][j]
                    self.round_keys[i].append(byte)

        # print(self.round_keys)
        return self.round_keys[40:44]

    def encrypt(self, plaintext):
        self.plain_state = text2matrix(plaintext)

        self.__add_round_key(self.plain_state, self.round_keys[:4])

        for i in range(1, 10):
            self.__round_encrypt(self.plain_state, self.round_keys[4 * i : 4 * (i + 1)])

        self.__sub_bytes(self.plain_state)
        self.__shift_rows(self.plain_state)
        self.__add_round_key(self.plain_state, self.round_keys[40:])

        return matrix2text(self.plain_state)

    def decrypt(self, ciphertext):
        self.cipher_state = text2matrix(ciphertext)

        self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.__inv_shift_rows(self.cipher_state)
        self.__inv_sub_bytes(self.cipher_state)

        for i in range(9, 0, -1):
            self.__round_decrypt(self.cipher_state, self.round_keys[4 * i : 4 * (i + 1)])

        self.__add_round_key(self.cipher_state, self.round_keys[:4])

        return matrix2text(self.cipher_state)

    def __add_round_key(self, s, k):
        for i in range(4):
            for j in range(4):
                s[i][j] ^= k[i][j]


    def __round_encrypt(self, state_matrix, key_matrix):
        self.__sub_bytes(state_matrix)
        self.__shift_rows(state_matrix)
        self.__mix_columns(state_matrix)
        self.__add_round_key(state_matrix, key_matrix)


    def __round_decrypt(self, state_matrix, key_matrix):
        self.__add_round_key(state_matrix, key_matrix)
        self.__inv_mix_columns(state_matrix)
        self.__inv_shift_rows(state_matrix)
        self.__inv_sub_bytes(state_matrix)

    def __sub_bytes(self, s):
        for i in range(4):
            for j in range(4):
                s[i][j] = self.faulty_sbox[s[i][j]]


    def __inv_sub_bytes(self, s):
        for i in range(4):
            for j in range(4):
                s[i][j] = self.inv_faulty_sbox[s[i][j]]


    def __shift_rows(self, s):
        s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]


    def __inv_shift_rows(self, s):
        s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

    def __mix_single_column(self, a):
        # please see Sec 4.1.2 in The Design of Rijndael
        t = a[0] ^ a[1] ^ a[2] ^ a[3]
        u = a[0]
        a[0] ^= t ^ xtime(a[0] ^ a[1])
        a[1] ^= t ^ xtime(a[1] ^ a[2])
        a[2] ^= t ^ xtime(a[2] ^ a[3])
        a[3] ^= t ^ xtime(a[3] ^ u)


    def __mix_columns(self, s):
        for i in range(4):
            self.__mix_single_column(s[i])


    def __inv_mix_columns(self, s):
        # see Sec 4.1.3 in The Design of Rijndael
        for i in range(4):
            u = xtime(xtime(s[i][0] ^ s[i][2]))
            v = xtime(xtime(s[i][1] ^ s[i][3]))
            s[i][0] ^= u
            s[i][1] ^= v
            s[i][2] ^= u
            s[i][3] ^= v

        self.__mix_columns(s)
```

## Using NIST Test Vectors to Verify the Correctness of AES Implementation
Reference: Test vectors have been taken from [NIST](https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/AES_Core128.pdf)


```python
master_key = 0x2B7E151628AED2A6ABF7158809CF4F3C
plaintext = 0x6BC1BEE22E409F96E93D7E117393172A
expected_ciphertext = 0x3AD77BB40D7A3660A89ECAF32466EF97
faulty_aes = AES(master_key)
faulty_aes.apply_fault(number_of_fault=0)
derived_ciphertext = faulty_aes.encrypt(plaintext)
if derived_ciphertext == expected_ciphertext:
    print("AES encryption works correctly :-)")
else:
    print("AES encryption doesn't work correctly :-(")
derived_plaintext = faulty_aes.decrypt(derived_ciphertext)
print("derived ciphertext:\t %s" % hex(derived_ciphertext).upper())
print("expected ciphertext:\t %s" % hex(expected_ciphertext).upper())
if derived_ciphertext == expected_ciphertext:
    print("AES decryption works correctly :-)")
else:
    print("AES decryption doesn't work correctly :-(")
print("derived plaintext:\t %s" % hex(derived_plaintext).upper())
print("expected plaintext:\t %s" % hex(plaintext).upper())
```

    AES encryption works correctly :-)
    derived ciphertext:	 0X3AD77BB40D7A3660A89ECAF32466EF97
    expected ciphertext:	 0X3AD77BB40D7A3660A89ECAF32466EF97
    AES decryption works correctly :-)
    derived plaintext:	 0X6BC1BEE22E409F96E93D7E117393172A
    expected plaintext:	 0X6BC1BEE22E409F96E93D7E117393172A


## Experiment 1
Assuming that `number_of_faults` faults have been induced, we generate `N` random (faulty) ciphertexts and collect observed values at each output byte of ciphertexts. We repeat this experiment for several random master keys and aiming to see how many random queries is required on average to observe all possible values at least once for an arbitrary output byte of ciphertext.


```python
def experiment1(number_of_experiments, number_of_faults, number_of_queries_in_each_exper):
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
```


```python
number_of_faults = 5
bias = 0
m = 2**8 - number_of_faults
expected_number_of_queries = int(np.ceil((m*harmonic_number(m)))) + bias
mean_output, total_mean = experiment1(number_of_experiments=10,\
                                      number_of_faults=number_of_faults,\
                                      number_of_queries_in_each_exper=expected_number_of_queries)
total_mean
```




    5.5687500000000005




```python
output = "Number of non-observed values on average:"
for row in range(4):
    output += "\n"
    for col in range(4):
        output += "%0.02f, " % mean_output[row][col]
plt.imshow(mean_output)
plt.xticks(range(4), range(4))
plt.yticks(range(4), range(4))
plt.colorbar()
plt.show()
print(output)
print('Number of faults: %d' % number_of_faults)
print('Expected number of queries based on our paper: %d' % expected_number_of_queries)
```


    
![png](output_19_0.png)
    


    Number of non-observed values on average:
    5.60, 5.60, 5.60, 5.20, 
    5.50, 5.40, 5.50, 5.70, 
    5.90, 5.80, 6.00, 5.40, 
    5.40, 5.10, 5.60, 5.80, 
    Number of faults: 5
    Expected number of queries based on our paper: 1533


**Produce Diagrams**


```python
def generate_digrams_data(number_of_faults=1,\
                          max_number_of_queries=1000,\
                          number_of_experiment_per_query=10):
    remained_deltas = []
    for nq in range(1, max_number_of_queries):
        mean_output, total_mean = experiment1(number_of_experiments=number_of_experiment_per_query,\
                                              number_of_faults=number_of_faults,\
                                              number_of_queries_in_each_exper=nq)
        remained_deltas.append(total_mean)
    return remained_deltas
```

**Generate (or Read) the Required Data to Plot a Figure for Number of Non-observed Values With Respect to the Number of Available Ciphertexts**


```python
lam = 1

m = 2**8 - lam
expected_number_of_queries = int(np.ceil((m*harmonic_number(m))))
max_number_of_queries = expected_number_of_queries + 800
#################################################################
number_of_faults = range(1, 17)
flag = 'read_data'
if flag == 'compute_data':
    number_of_experiment_per_query = 10
    with Pool(16) as pool:
        arguments = [(nf, max_number_of_queries, number_of_experiment_per_query) for nf in number_of_faults]
        candidates = pool.starmap(generate_digrams_data, arguments)
    with open('candidates', 'wb') as f:
        pickle.dump(candidates, f)
elif flag == 'read_data':
    with open('candidates', 'rb') as f:
        candidates = pickle.load(f)
```

**The following vector includes the average number of non-observed values when the number of faults is equal to 1 and N known ciphertexts are available, where 1 <= N <= 2360:**



```python
y_data_1 = candidates[0]
print(y_data_1)
```

    [255.0, 254.00187500000004, 253.01125000000005, 252.020625, 251.044375, 250.0525, 249.0725, 248.116875, 247.1575, 246.1925, 245.21249999999998, 244.25624999999997, 243.31124999999997, 242.365625, 241.41187500000004, 240.44875, 239.52625, 238.600625, 237.65312500000002, 236.73374999999996, 235.823125, 234.853125, 233.94937499999995, 233.06187500000001, 232.148125, 231.23125, 230.303125, 229.49249999999998, 228.52937500000002, 227.64687500000002, 226.76937499999997, 225.95937500000002, 224.98687500000003, 224.080625, 223.258125, 222.34375, 221.54625000000004, 220.63500000000005, 219.820625, 218.97375000000002, 218.08625, 217.26437499999997, 216.36499999999998, 215.51187500000003, 214.614375, 213.80375, 213.00875, 212.22187499999998, 211.33875000000003, 210.556875, 209.718125, 208.85937500000003, 207.97250000000003, 207.23250000000002, 206.47062500000004, 205.62937499999998, 204.839375, 204.151875, 203.24375000000003, 202.42187499999997, 201.73624999999998, 200.840625, 200.02437500000002, 199.366875, 198.49812500000002, 197.70749999999998, 197.0475, 196.21687500000002, 195.37875000000003, 194.72250000000003, 193.96499999999995, 193.0825, 192.38062499999998, 191.6675, 190.96687500000002, 190.194375, 189.48624999999998, 188.6925, 187.9525, 187.20187500000003, 186.421875, 185.80562500000002, 184.885, 184.20875000000004, 183.78375, 182.904375, 182.25374999999997, 181.52999999999997, 180.86, 180.01500000000001, 179.47062500000004, 178.74, 177.96, 177.295625, 176.67812500000002, 175.83812500000002, 175.22250000000005, 174.43125, 174.12875, 173.07375000000002, 172.5875, 171.795, 171.19312499999998, 170.54874999999998, 169.805, 169.264375, 168.500625, 167.93187500000002, 167.28624999999997, 166.556875, 165.845, 165.313125, 164.58562500000002, 163.836875, 163.224375, 162.780625, 162.06687499999998, 161.440625, 160.78187499999999, 160.18125, 159.613125, 158.95250000000001, 158.369375, 157.465, 157.10625, 156.47812499999995, 155.925, 155.5375, 154.67312499999997, 154.10625, 153.42062499999997, 152.7475, 152.069375, 151.65375, 151.0225, 150.545625, 149.854375, 149.46562500000005, 148.68999999999997, 148.21625, 147.68312500000002, 146.92125, 146.37000000000003, 145.901875, 145.39374999999998, 144.635, 144.156875, 143.6875, 143.0, 142.43562500000002, 142.16062499999998, 141.430625, 140.72062499999998, 140.40625000000003, 139.8425, 139.03124999999997, 138.64749999999998, 138.185625, 137.66625, 137.010625, 136.393125, 135.975625, 135.409375, 134.7325, 134.506875, 133.876875, 133.33499999999998, 132.935, 132.381875, 131.93124999999998, 131.194375, 130.73812500000003, 130.39875, 129.73000000000002, 129.199375, 128.89, 128.06625, 127.88999999999999, 127.41875, 126.74875, 126.20562500000001, 125.809375, 125.4025, 124.89874999999999, 124.37437499999999, 123.86437499999998, 123.33062500000001, 122.795, 122.53812500000001, 121.86125000000003, 121.68124999999998, 121.08187499999998, 120.583125, 120.16187500000001, 119.70875000000002, 119.21125, 118.633125, 118.14437500000001, 117.88062500000001, 117.309375, 116.97437500000001, 116.38312499999999, 115.85875, 115.408125, 115.255, 114.640625, 114.2425, 113.894375, 113.174375, 112.7575, 112.360625, 111.89187499999998, 111.7, 111.10999999999999, 110.674375, 110.25625, 109.83875, 109.40312499999999, 109.08937499999999, 108.59875, 107.94562499999999, 107.7425, 107.43937500000001, 106.79625000000001, 106.40187499999999, 106.215625, 105.715, 105.40124999999999, 104.944375, 104.328125, 103.89312500000003, 103.875625, 103.11687500000002, 102.95625000000001, 102.31937500000001, 101.875, 101.3825, 100.89312500000001, 100.78875000000001, 100.233125, 100.16, 99.83999999999997, 99.08749999999998, 98.95812499999998, 98.41437499999999, 98.009375, 97.72687499999999, 97.364375, 96.89375, 96.70187499999999, 96.295625, 95.87812499999998, 95.56937500000001, 95.178125, 94.71874999999999, 94.329375, 93.90687500000001, 93.795625, 93.48, 93.28312500000001, 92.55187500000001, 92.29749999999999, 91.88000000000001, 91.65374999999999, 91.14437500000003, 90.94187500000001, 90.2875, 89.9825, 89.66562499999998, 89.54499999999999, 89.03125, 88.690625, 88.32187499999998, 88.006875, 87.86312499999998, 87.33437500000001, 87.10125000000001, 86.776875, 86.296875, 86.00250000000001, 85.61812499999999, 85.32875, 84.92374999999998, 84.73749999999998, 84.486875, 84.04375, 83.71499999999999, 83.53687500000001, 83.04625, 82.81, 82.49000000000001, 81.87249999999999, 81.731875, 81.22375, 81.405, 80.845625, 80.4, 80.321875, 79.998125, 79.40687500000001, 79.33312500000001, 78.954375, 78.673125, 78.4025, 78.07125000000002, 77.859375, 77.439375, 77.09875, 77.04, 76.61625, 76.29, 75.993125, 75.58437500000001, 75.35687499999999, 75.193125, 74.911875, 74.66187500000001, 74.129375, 73.87124999999999, 73.764375, 73.45125, 73.00375, 73.058125, 72.626875, 72.268125, 71.85875000000001, 71.71375, 71.42187500000001, 71.174375, 70.88687499999999, 70.57999999999998, 70.42875, 70.27687499999999, 69.910625, 69.610625, 69.22000000000001, 68.976875, 68.646875, 68.4675, 68.2125, 67.76875000000001, 67.6125, 67.52375, 67.18562499999999, 66.848125, 66.47625, 66.38062500000001, 66.044375, 65.928125, 65.56750000000001, 65.223125, 64.99187500000001, 64.97124999999998, 64.53000000000002, 64.25875, 64.11625, 63.91812500000001, 63.868125, 63.33562500000001, 63.098124999999996, 62.794375, 62.500000000000014, 62.33, 62.175, 61.9575, 61.71124999999999, 61.565625000000004, 61.233125, 60.9025, 60.718125, 60.376875000000005, 60.17, 60.077500000000015, 59.79999999999999, 59.49875, 59.47624999999999, 59.36999999999999, 58.91312500000001, 58.96187499999999, 58.22687500000001, 58.353124999999984, 57.921875, 57.844375, 57.520624999999995, 57.183125000000004, 57.19125, 56.856875, 56.65375, 56.435, 56.32187500000001, 55.839375, 56.043125, 55.67687499999999, 55.633125, 55.18625, 54.94437500000001, 54.39374999999998, 54.461250000000014, 54.15124999999999, 54.376875000000005, 53.92249999999999, 53.623125, 53.614999999999995, 53.34, 52.93187499999999, 52.933749999999996, 52.597500000000004, 52.286249999999995, 52.12375, 51.96187499999999, 51.746249999999996, 51.708124999999995, 51.555625000000006, 51.31, 51.00749999999999, 50.895624999999995, 50.499375, 50.56187499999999, 50.176249999999996, 50.10187500000001, 49.881874999999994, 49.766875, 49.67124999999999, 49.353750000000005, 49.20312500000001, 48.88749999999999, 48.669999999999995, 48.512499999999996, 48.291875000000005, 48.28937500000001, 47.70187499999999, 47.721875, 47.77374999999999, 47.381874999999994, 47.260625, 47.38562499999999, 46.99187500000001, 46.718125, 46.450624999999995, 46.55875, 46.32, 46.11999999999999, 45.92875, 45.66374999999999, 45.552499999999995, 45.41125000000001, 45.439375000000005, 44.985, 44.80249999999999, 44.73062499999999, 44.73875000000001, 44.193125, 44.113125000000004, 44.131875, 43.64875, 43.503125, 43.716875, 43.326875, 43.11687500000001, 43.003125, 43.080625000000005, 42.55, 42.279375, 42.284375000000004, 42.161249999999995, 42.081875, 41.91625, 41.612500000000004, 41.51937500000001, 41.356875, 41.03187500000001, 41.011874999999996, 40.792500000000004, 40.670625, 40.79624999999999, 40.5025, 40.345625, 40.09375, 39.98, 39.826875, 39.74375, 39.610625, 39.4225, 39.335, 39.12125, 38.92875, 38.930625, 38.433125, 38.525625, 38.110625, 38.058749999999996, 38.243125000000006, 37.942499999999995, 37.78750000000001, 37.63875, 37.40812499999999, 37.228125000000006, 37.30875, 37.20687499999999, 36.99875, 36.78062500000001, 36.558125000000004, 36.619375, 36.4, 36.135, 36.178124999999994, 36.12, 35.9, 35.66499999999999, 35.651250000000005, 35.40875, 35.350624999999994, 35.001875, 34.94125, 34.830625, 34.49875, 34.77, 34.50125, 34.194375, 34.348125, 34.058125000000004, 33.99, 33.79625, 33.705625, 33.573750000000004, 33.271875, 33.336875, 33.318125, 33.066874999999996, 32.9775, 32.775000000000006, 32.791875000000005, 32.63, 32.243750000000006, 32.290625, 32.02375000000001, 31.9125, 31.946875000000006, 31.864375000000003, 31.666249999999998, 31.678125, 31.562499999999996, 31.431249999999995, 31.181875, 31.079375, 30.874375, 30.723125, 30.6225, 30.66625, 30.448750000000004, 30.398124999999993, 30.558750000000003, 30.25625, 30.084374999999998, 30.006874999999997, 30.085, 29.746249999999996, 29.695, 29.604999999999997, 29.298125, 29.379375000000007, 29.140625000000007, 28.981875000000002, 28.954375, 28.808125, 28.885, 28.714375, 28.453125, 28.229374999999997, 28.336874999999996, 28.133125, 28.07, 28.055000000000007, 27.868125, 27.6825, 27.761874999999996, 27.3375, 27.437499999999996, 27.310625, 27.126875, 27.3575, 26.986874999999998, 27.085625, 26.9525, 26.634375, 26.440625000000004, 26.614375, 26.479999999999997, 26.34375, 26.155, 26.194375, 25.885625, 25.980625, 25.875, 25.75875, 25.645625000000003, 25.34375, 25.368750000000006, 25.393125, 25.305000000000003, 25.441874999999992, 25.050625, 25.11875, 24.971875, 24.811874999999997, 24.6825, 24.591875, 24.405000000000005, 24.355625, 24.395624999999995, 24.143749999999997, 24.000625, 24.016875, 23.821875, 23.854375000000005, 23.799375, 23.589999999999996, 23.4, 23.508125, 23.424375, 23.643124999999998, 23.07875, 23.343750000000004, 23.106875, 23.243749999999995, 22.849375000000002, 22.912499999999998, 22.875624999999996, 22.769375, 22.666875, 22.503125, 22.41375, 22.323749999999997, 22.338749999999997, 22.147500000000004, 22.053124999999998, 22.065624999999997, 22.083750000000002, 21.923125, 21.819375, 21.665, 21.674375, 21.402499999999996, 21.428125, 21.345000000000002, 21.272499999999997, 21.297500000000003, 21.238125, 21.131874999999997, 20.966250000000002, 20.964375, 20.67625, 20.769999999999996, 20.58625, 20.765624999999996, 20.488749999999996, 20.469375000000003, 20.210624999999997, 20.437500000000004, 20.098125, 20.185000000000002, 20.229999999999997, 19.940000000000005, 19.895, 19.83625, 19.725, 19.738125, 19.718124999999997, 19.589375, 19.457500000000003, 19.349375, 19.34125, 19.330000000000002, 19.097500000000004, 19.1925, 19.18375, 18.820625, 18.935624999999998, 18.729999999999997, 18.872500000000002, 18.916874999999997, 18.581875000000004, 18.671875, 18.45375, 18.4375, 18.266875, 18.123749999999998, 18.135625, 18.07875, 18.067500000000003, 17.986875, 17.8575, 17.900624999999998, 17.834999999999997, 17.80625, 17.73125, 17.594375, 17.654375, 17.381875, 17.347500000000004, 17.334375, 17.245, 17.281875, 17.22, 17.259375, 16.950625, 16.775624999999998, 16.893124999999998, 16.95125, 16.745624999999997, 16.773125, 16.659375, 16.464375, 16.643124999999998, 16.45, 16.42625, 16.4375, 16.2525, 16.194374999999997, 16.24375, 16.095, 15.98625, 15.883125, 15.841875000000002, 15.770624999999997, 15.973125, 15.88125, 15.723749999999999, 15.590625, 15.630625, 15.531875000000001, 15.563125, 15.476875, 15.247499999999999, 15.311875, 15.3875, 15.4225, 15.190625, 14.996875, 15.139375000000001, 14.856874999999997, 14.92625, 14.865625000000001, 14.749375, 14.7575, 14.543125, 14.636875, 14.690624999999999, 14.602500000000001, 14.453125, 14.442499999999999, 14.412500000000001, 14.420625000000001, 14.241874999999999, 14.103750000000002, 14.284375, 14.198750000000002, 13.953125, 14.090625000000001, 13.995000000000001, 13.948749999999999, 13.780000000000001, 13.87875, 13.824374999999998, 13.8775, 13.5625, 13.636249999999997, 13.604999999999999, 13.552500000000002, 13.474374999999998, 13.509374999999999, 13.413749999999997, 13.356875, 13.283125000000002, 13.34, 13.208125, 13.031250000000002, 13.1, 12.985, 13.031875000000003, 12.779375, 12.82375, 12.8375, 12.7425, 12.66625, 12.605624999999998, 12.815624999999999, 12.61625, 12.531249999999998, 12.628125, 12.461875, 12.334375, 12.406875000000001, 12.288124999999999, 12.26875, 12.371875000000001, 12.317499999999999, 12.238749999999998, 12.166875000000001, 12.126874999999998, 12.120625000000002, 11.925625, 11.862499999999999, 11.849375000000002, 11.950625, 11.875, 11.756875, 11.805625, 11.631875, 11.585625, 11.6275, 11.514375000000001, 11.526250000000001, 11.3875, 11.626249999999999, 11.491875, 11.338125000000002, 11.293125000000002, 11.307499999999997, 11.25875, 11.227499999999997, 11.071875000000002, 11.010625, 11.075625, 11.075624999999999, 11.005624999999998, 10.969375, 10.956875000000002, 10.865625, 10.808125, 10.955625, 10.890624999999998, 10.788125, 10.8075, 10.81, 10.657499999999999, 10.661875000000002, 10.590625, 10.535, 10.6025, 10.5025, 10.458125, 10.44125, 10.23625, 10.405000000000001, 10.349375, 10.11125, 10.236875, 10.1625, 9.966249999999999, 10.051874999999997, 10.030624999999999, 10.186875, 9.83375, 9.978750000000002, 10.0075, 9.87375, 9.705625, 9.818750000000001, 9.67375, 9.66625, 9.65625, 9.594375, 9.6375, 9.5625, 9.529375000000002, 9.381874999999999, 9.619375, 9.520625, 9.404375, 9.413124999999999, 9.45875, 9.309375000000001, 9.32625, 9.374375, 9.219999999999997, 9.309375, 9.131875, 9.12875, 9.11375, 9.124375000000002, 9.04875, 8.976875, 8.929375000000002, 8.941875, 9.066875, 8.837499999999999, 8.771875000000001, 8.891875, 8.725625, 8.73375, 8.821250000000001, 8.815624999999999, 8.662500000000001, 8.773125, 8.628124999999999, 8.589374999999999, 8.590625, 8.540625000000002, 8.55625, 8.459375, 8.51375, 8.373125, 8.28875, 8.325625000000002, 8.31125, 8.30875, 8.25, 8.250625000000001, 8.1475, 8.141874999999999, 8.2, 8.15375, 8.05625, 8.04375, 8.055625000000001, 7.93875, 7.987500000000001, 7.949375, 8.06875, 7.920624999999999, 7.858125, 7.810624999999999, 7.916250000000002, 7.6825, 7.795625000000001, 7.784375, 7.694375, 7.699999999999999, 7.7250000000000005, 7.609374999999998, 7.5775000000000015, 7.571249999999999, 7.566250000000001, 7.486875, 7.456249999999999, 7.48, 7.45875, 7.49375, 7.458125000000002, 7.37625, 7.26625, 7.353125000000001, 7.330625, 7.300625, 7.310624999999998, 7.327500000000001, 7.222500000000001, 7.258749999999999, 7.123124999999999, 7.098749999999998, 7.195, 7.083124999999999, 7.034375, 7.028125000000001, 7.01375, 7.039375000000001, 6.9606249999999985, 6.834374999999999, 7.009375, 6.951250000000001, 6.851249999999999, 6.883750000000001, 6.807499999999999, 6.833749999999999, 6.79625, 6.753749999999999, 6.826875000000001, 6.795, 6.706875, 6.768750000000001, 6.753750000000001, 6.621875, 6.58625, 6.63875, 6.6850000000000005, 6.538749999999999, 6.541250000000001, 6.4468749999999995, 6.493749999999999, 6.52625, 6.42, 6.508750000000001, 6.490625, 6.499375000000001, 6.41625, 6.32625, 6.3618749999999995, 6.317500000000001, 6.3025, 6.36125, 6.262500000000001, 6.181875000000002, 6.200000000000001, 6.200000000000001, 6.210624999999999, 6.16, 6.116249999999998, 5.958750000000001, 6.005000000000002, 6.05625, 5.972499999999999, 6.022500000000001, 5.985625000000001, 5.994374999999999, 5.936875000000001, 5.876249999999999, 5.939375, 5.936249999999999, 5.860625000000001, 5.816875, 5.874375000000001, 5.940624999999999, 5.733750000000001, 5.789999999999999, 5.802500000000002, 5.6156250000000005, 5.658125, 5.7175, 5.726874999999999, 5.649375, 5.62625, 5.616249999999999, 5.645625000000001, 5.636875000000001, 5.609999999999999, 5.502499999999999, 5.570625000000001, 5.461250000000001, 5.4575, 5.57125, 5.4556249999999995, 5.445000000000001, 5.4818750000000005, 5.415000000000001, 5.480625000000001, 5.426875000000001, 5.41375, 5.359375, 5.44875, 5.403750000000001, 5.16, 5.2725, 5.274374999999999, 5.220625, 5.118125, 5.298125000000001, 5.2787500000000005, 5.1525, 5.178125, 5.133749999999999, 5.07125, 5.0443750000000005, 5.080625, 5.11875, 5.043125, 5.056875000000001, 5.04375, 5.0337499999999995, 5.0725, 4.925625, 4.893125000000001, 4.93125, 4.946875, 4.9037500000000005, 4.876250000000001, 4.875000000000001, 4.936875000000001, 4.864375, 4.934375, 4.829375000000001, 4.8374999999999995, 4.85, 4.809375, 4.836874999999999, 4.716875, 4.728125000000001, 4.768125, 4.70625, 4.6206249999999995, 4.5825000000000005, 4.6225000000000005, 4.599375, 4.7, 4.5975, 4.6137500000000005, 4.616875, 4.576875, 4.545, 4.610625000000001, 4.581874999999999, 4.555, 4.585625, 4.4487499999999995, 4.5225, 4.4506250000000005, 4.468125, 4.475625, 4.4937499999999995, 4.449999999999999, 4.42, 4.336250000000001, 4.38875, 4.45125, 4.421875, 4.364999999999999, 4.363125, 4.261875, 4.306875, 4.35125, 4.336875, 4.26125, 4.273125, 4.324375, 4.26, 4.19125, 4.288125000000001, 4.128125, 4.160625, 4.1575, 4.149375, 4.124375, 4.043125, 4.081250000000001, 4.070625, 4.115, 4.1025, 4.01875, 4.0131250000000005, 4.059374999999999, 4.018125, 4.054375, 3.9699999999999998, 4.07125, 3.9656249999999997, 3.9581250000000003, 4.064375, 3.9318749999999993, 3.9225, 3.9375, 3.9937499999999995, 3.88875, 3.901250000000001, 3.8537499999999993, 3.8468749999999994, 3.81625, 3.8368750000000005, 3.818125, 3.7325000000000004, 3.8474999999999997, 3.818125, 3.7700000000000005, 3.8206250000000006, 3.80875, 3.8374999999999995, 3.706875, 3.7049999999999996, 3.7218750000000007, 3.68625, 3.7450000000000006, 3.699375, 3.630625, 3.706875, 3.6031250000000004, 3.600625, 3.5906250000000006, 3.5931250000000006, 3.562500000000001, 3.660625, 3.689375, 3.6250000000000004, 3.6293750000000005, 3.5718749999999995, 3.5825, 3.5168749999999998, 3.6106249999999993, 3.5968750000000003, 3.4887499999999996, 3.599375, 3.4850000000000003, 3.4612499999999993, 3.4749999999999996, 3.4987499999999994, 3.4818750000000005, 3.4249999999999994, 3.500625, 3.4262499999999996, 3.4050000000000002, 3.37875, 3.339375, 3.371875, 3.4206250000000002, 3.3487499999999994, 3.375625, 3.2818750000000003, 3.3525, 3.3925, 3.3837500000000005, 3.34, 3.30125, 3.31375, 3.2818750000000003, 3.295, 3.34375, 3.2381250000000006, 3.264375, 3.239375, 3.2550000000000003, 3.2374999999999994, 3.1975000000000002, 3.1650000000000005, 3.1937499999999996, 3.195, 3.243125, 3.169375, 3.1899999999999995, 3.1525000000000003, 3.1287499999999997, 3.2181250000000006, 3.0912499999999996, 3.216875, 3.0968750000000003, 3.1106249999999998, 3.190625, 3.063125, 3.0549999999999997, 3.09875, 3.1012500000000003, 3.0675000000000003, 3.0881250000000002, 3.0700000000000007, 3.0387499999999994, 3.05, 3.0693750000000004, 3.0293750000000004, 2.96375, 2.98875, 3.0618749999999997, 2.9456250000000006, 2.989375, 2.97875, 2.9718750000000003, 2.9687499999999996, 2.96375, 2.9368750000000006, 2.9624999999999995, 2.9299999999999997, 2.9224999999999994, 2.9206250000000002, 2.87, 2.885625, 2.865, 2.946875, 2.9118749999999998, 2.89125, 2.871875, 2.880625, 2.8250000000000006, 2.9106250000000005, 2.8712500000000003, 2.7849999999999997, 2.791875, 2.7868749999999993, 2.765625, 2.81875, 2.78, 2.799375, 2.7893749999999997, 2.775, 2.7268749999999997, 2.7600000000000002, 2.775625, 2.7493749999999997, 2.7431249999999996, 2.701250000000001, 2.706875, 2.694375, 2.65125, 2.6725, 2.6981250000000006, 2.6618749999999998, 2.689375, 2.6881250000000003, 2.6706250000000002, 2.6925, 2.6818750000000002, 2.65875, 2.6562500000000004, 2.609375, 2.581875, 2.6075000000000004, 2.6012499999999994, 2.605625, 2.601875, 2.59375, 2.6087499999999997, 2.549375, 2.6262499999999998, 2.581875, 2.524375, 2.543125, 2.536250000000001, 2.5456250000000007, 2.5118750000000003, 2.49125, 2.5181249999999995, 2.4975, 2.5225, 2.4981250000000004, 2.555625, 2.5193749999999997, 2.555625, 2.4406250000000003, 2.4475, 2.4999999999999996, 2.4225000000000003, 2.4256249999999997, 2.5056249999999993, 2.453125, 2.470625, 2.3831249999999997, 2.480625, 2.4593749999999996, 2.404375, 2.3781250000000003, 2.39625, 2.491875, 2.3825000000000003, 2.4368749999999997, 2.41625, 2.376875, 2.3549999999999995, 2.3856249999999997, 2.3200000000000003, 2.3862499999999995, 2.3887500000000004, 2.3537500000000002, 2.36625, 2.35, 2.325625, 2.3193750000000004, 2.31, 2.33625, 2.319375, 2.2575, 2.3181249999999998, 2.336875, 2.301875, 2.256875, 2.3000000000000003, 2.2987500000000005, 2.27, 2.26375, 2.234375, 2.2712499999999993, 2.2068750000000006, 2.285, 2.243125, 2.2725, 2.244375, 2.254375, 2.21125, 2.178125, 2.1831249999999995, 2.195625, 2.1756249999999997, 2.21875, 2.2075, 2.174375, 2.178125, 2.1662500000000002, 2.223125, 2.21125, 2.17375, 2.173125, 2.1575, 2.1725, 2.1318750000000004, 2.131875, 2.195, 2.1725, 2.1762500000000005, 2.1381250000000005, 2.1443749999999997, 2.109375, 2.15875, 2.1106249999999998, 2.09375, 2.135, 2.0693749999999995, 2.144375, 2.083125, 2.051875, 2.0475, 2.0962499999999995, 2.043125, 2.101875, 2.0681249999999998, 2.110625, 2.088125, 2.074375, 2.0637499999999998, 2.0531249999999996, 2.0212499999999998, 2.0187500000000003, 2.0418749999999997, 2.0318750000000003, 2.019375, 2.0331249999999996, 1.9775, 2.005625, 2.0181250000000004, 1.9900000000000002, 1.989375, 1.974375, 1.985625, 1.9962499999999996, 1.9725, 2.03625, 1.9443749999999997, 1.9643749999999998, 1.975625, 1.9381249999999999, 1.961875, 2.0156249999999996, 1.951875, 1.9500000000000002, 1.9843749999999998, 1.945625, 1.94125, 1.9587499999999998, 1.9087500000000002, 1.9224999999999999, 1.9018749999999998, 1.9031250000000004, 1.914375, 1.946875, 1.92625, 1.888125, 1.9093749999999998, 1.910625, 1.8712499999999999, 1.8981249999999998, 1.9187500000000002, 1.86125, 1.89875, 1.8712499999999996, 1.87875, 1.8712499999999999, 1.8375, 1.875, 1.895, 1.8662500000000004, 1.90875, 1.8875000000000002, 1.878125, 1.8775, 1.8775, 1.8575000000000002, 1.8675, 1.87875, 1.8662500000000004, 1.83375, 1.79875, 1.834375, 1.81, 1.8031250000000003, 1.7806250000000001, 1.8337500000000004, 1.8162500000000001, 1.7662499999999999, 1.8087499999999999, 1.813125, 1.814375, 1.8037500000000002, 1.7825, 1.7999999999999998, 1.775625, 1.785, 1.7587500000000003, 1.713125, 1.775, 1.7793750000000002, 1.71375, 1.7775, 1.733125, 1.7362499999999998, 1.736875, 1.7512499999999998, 1.75, 1.7325000000000002, 1.7687499999999998, 1.7049999999999998, 1.7181250000000001, 1.7206250000000003, 1.7212500000000004, 1.7168750000000004, 1.6943750000000002, 1.6837499999999999, 1.733125, 1.710625, 1.7212500000000002, 1.6968749999999997, 1.7262499999999998, 1.6931249999999998, 1.7112500000000002, 1.6924999999999997, 1.739375, 1.6650000000000003, 1.7018749999999998, 1.686875, 1.695625, 1.658125, 1.664375, 1.6399999999999997, 1.6625, 1.65, 1.6768750000000001, 1.6949999999999998, 1.6418749999999998, 1.641875, 1.6343750000000001, 1.66625, 1.6481249999999996, 1.653125, 1.6331250000000002, 1.6606250000000002, 1.630625, 1.644375, 1.63875, 1.6199999999999999, 1.6387499999999997, 1.6443749999999997, 1.6443749999999997, 1.5912499999999998, 1.6006250000000002, 1.626875, 1.6368749999999999, 1.6049999999999998, 1.628125, 1.5831250000000001, 1.6175000000000002, 1.6206250000000002, 1.6181249999999998, 1.62125, 1.5887499999999999, 1.5762500000000002, 1.5956250000000003, 1.606875, 1.581875, 1.5931250000000003, 1.61625, 1.573125, 1.58, 1.5781249999999998, 1.59125, 1.55, 1.5875, 1.5675, 1.5343749999999998, 1.549375, 1.5262499999999999, 1.5662500000000001, 1.548125, 1.5468749999999998, 1.5743749999999999, 1.5612500000000002, 1.5525000000000002, 1.5325000000000002, 1.535625, 1.535625, 1.56625, 1.5575, 1.545, 1.5268750000000002, 1.515, 1.5762500000000002, 1.5387500000000003, 1.5437500000000002, 1.4968749999999995, 1.49, 1.52125, 1.5099999999999998, 1.49375, 1.52, 1.498125, 1.510625, 1.493125, 1.4837500000000001, 1.4812500000000002, 1.49125, 1.4662499999999998, 1.535, 1.4825000000000002, 1.523125, 1.53375, 1.5156250000000002, 1.508125, 1.48875, 1.463125, 1.4900000000000002, 1.504375, 1.459375, 1.4537499999999997, 1.4774999999999998, 1.45375, 1.4493749999999999, 1.4550000000000003, 1.461875, 1.495625, 1.4687499999999998, 1.4587500000000002, 1.464375, 1.4768750000000002, 1.4256250000000001, 1.45375, 1.460625, 1.44125, 1.4512499999999997, 1.414375, 1.4512500000000002, 1.425, 1.448125, 1.416875, 1.4581250000000001, 1.435625, 1.4393749999999998, 1.415625, 1.4412500000000001, 1.4231250000000002, 1.4175, 1.4224999999999999, 1.4118749999999998, 1.4412499999999997, 1.4381249999999997, 1.419375, 1.4337499999999999, 1.413125, 1.41, 1.4412499999999997, 1.44625, 1.4118750000000004, 1.398125, 1.4143750000000002, 1.414375, 1.393125, 1.4112500000000001, 1.4175, 1.404375, 1.38375, 1.3862500000000002, 1.384375, 1.3768749999999996, 1.384375, 1.395, 1.36375, 1.354375, 1.379375, 1.3949999999999998, 1.405, 1.3768750000000003, 1.3737499999999998, 1.3756249999999999, 1.390625, 1.3774999999999997, 1.381875, 1.3525, 1.3699999999999999, 1.3450000000000002, 1.3699999999999997, 1.3525000000000003, 1.374375, 1.358125, 1.3456249999999998, 1.345625, 1.3662500000000002, 1.3712499999999999, 1.345625, 1.361875, 1.408125, 1.3750000000000002, 1.3406250000000002, 1.3418750000000002, 1.328125, 1.3549999999999998, 1.349375, 1.3581250000000002, 1.36625, 1.35125, 1.3268750000000002, 1.325, 1.3462500000000002, 1.3718750000000002, 1.3431250000000001, 1.3406250000000002, 1.3318750000000001, 1.32, 1.3256249999999998, 1.31625, 1.3243749999999999, 1.345, 1.3212499999999998, 1.313125, 1.3356249999999998, 1.368125, 1.3356250000000003, 1.3031249999999996, 1.341875, 1.331875, 1.3231249999999999, 1.3193750000000002, 1.3175000000000003, 1.31375, 1.3156249999999998, 1.3006250000000001, 1.304375, 1.31375, 1.29875, 1.306875, 1.2725, 1.2712500000000002, 1.2937500000000002, 1.3018750000000001, 1.3187499999999999, 1.2881250000000002, 1.3231249999999999, 1.3025, 1.2575, 1.2825, 1.2774999999999999, 1.28875, 1.288125, 1.300625, 1.2843750000000003, 1.28375, 1.26625, 1.293125, 1.2906250000000001, 1.2768749999999998, 1.295625, 1.259375, 1.285, 1.254375, 1.264375, 1.27625, 1.2575, 1.295, 1.289375, 1.28375, 1.2425, 1.27375, 1.251875, 1.2574999999999998, 1.241875, 1.285625, 1.249375, 1.2781249999999997, 1.2668750000000002, 1.273125, 1.2581250000000002, 1.2650000000000001, 1.2356249999999998, 1.280625, 1.2556249999999998, 1.2512500000000002, 1.2362499999999998, 1.261875, 1.2537500000000001, 1.2331249999999998, 1.26125, 1.261875, 1.2499999999999998, 1.2574999999999998, 1.22375, 1.2412500000000002, 1.23, 1.23625, 1.2418749999999998, 1.24125, 1.234375, 1.2331249999999998, 1.226875, 1.235, 1.2375, 1.2337500000000003, 1.2162499999999998, 1.2425, 1.246875, 1.23375, 1.23, 1.249375, 1.2081249999999997, 1.2275, 1.2275, 1.23125, 1.2425, 1.2275, 1.21875, 1.2118749999999998, 1.221875, 1.2262499999999998, 1.2225000000000001, 1.2118749999999998, 1.225, 1.2037499999999999, 1.209375, 1.2274999999999998, 1.215, 1.2181250000000001, 1.214375, 1.2006250000000003, 1.2324999999999997, 1.2199999999999998, 1.19875, 1.206875, 1.2025000000000001, 1.21125, 1.2181250000000001, 1.2025000000000001, 1.2200000000000002, 1.1918749999999998, 1.19875, 1.1987499999999998, 1.185625, 1.2200000000000002, 1.209375, 1.193125, 1.21875, 1.1843750000000002, 1.201875, 1.2075, 1.2218750000000003, 1.18875, 1.19375, 1.1868749999999997, 1.19625, 1.1824999999999999, 1.1843749999999997, 1.1881249999999999, 1.19375, 1.190625, 1.196875, 1.179375, 1.1925000000000001, 1.1862500000000002, 1.1675, 1.1743750000000002, 1.18, 1.1812500000000001, 1.17875, 1.2075, 1.1925000000000001, 1.168125, 1.1806249999999998, 1.1749999999999998, 1.1706249999999998, 1.1700000000000002, 1.1925, 1.171875, 1.17875, 1.1725, 1.1793749999999998, 1.1531250000000002, 1.1543749999999997, 1.1956250000000002, 1.1893749999999998, 1.176875, 1.1749999999999998, 1.1712500000000001, 1.1593749999999998, 1.1649999999999998, 1.165625, 1.17875, 1.1675, 1.178125, 1.17375, 1.1624999999999999, 1.15625, 1.164375, 1.1512499999999999, 1.156875, 1.1575, 1.1718749999999998, 1.1606249999999998, 1.1575, 1.15875, 1.1562499999999998, 1.17875, 1.1443750000000001, 1.1600000000000001, 1.15, 1.1525, 1.1524999999999999, 1.1587500000000002, 1.1443750000000001, 1.1531249999999997, 1.1693749999999998, 1.146875, 1.15875, 1.14875, 1.1525, 1.143125, 1.145625, 1.139375, 1.1512499999999997, 1.1549999999999996, 1.1243750000000001, 1.15375, 1.14625, 1.1418750000000002, 1.138125, 1.123125, 1.1381249999999998, 1.1306249999999998, 1.138125, 1.1612500000000001, 1.1456250000000001, 1.1425000000000003, 1.155625, 1.130625, 1.1218749999999997, 1.1375, 1.1443750000000001, 1.12875, 1.1381249999999998, 1.145625, 1.1349999999999998, 1.134375, 1.130625, 1.1225, 1.1275, 1.1325, 1.1337499999999998, 1.1225, 1.120625, 1.129375, 1.1343750000000001, 1.128125, 1.128125, 1.1318749999999997, 1.1306249999999998, 1.1225, 1.1212499999999999, 1.129375, 1.1168749999999998, 1.113125, 1.1231250000000002, 1.1243750000000001, 1.1275, 1.12125, 1.128125, 1.1175, 1.119375, 1.11, 1.1237499999999998, 1.134375, 1.134375, 1.1175, 1.11, 1.1324999999999998, 1.114375, 1.1356249999999999, 1.1124999999999998, 1.1218750000000004, 1.115625, 1.123125, 1.12, 1.11375, 1.1125000000000003, 1.11, 1.1143750000000001, 1.0981249999999998, 1.11625, 1.11125, 1.118125, 1.106875, 1.1075, 1.124375, 1.1081249999999998, 1.109375, 1.114375, 1.1075, 1.1181249999999998, 1.1075000000000002, 1.1168749999999998, 1.10125, 1.1006250000000002, 1.0987500000000001, 1.105625, 1.1012500000000003, 1.105625, 1.1025, 1.100625, 1.1018750000000002, 1.1050000000000002, 1.108125, 1.0950000000000002, 1.101875, 1.1025, 1.10625, 1.1075, 1.0956250000000003, 1.1018749999999997, 1.104375, 1.10375, 1.090625, 1.0975, 1.095625, 1.09125, 1.095, 1.0962500000000002, 1.0837500000000002, 1.089375, 1.1056249999999999, 1.0912500000000003, 1.088125, 1.093125, 1.0843750000000003, 1.1025, 1.0975, 1.0787499999999999, 1.0862500000000002, 1.0975000000000001, 1.0981249999999998, 1.084375, 1.091875, 1.0925000000000002, 1.085625, 1.0893749999999998, 1.09, 1.09375, 1.085625, 1.090625, 1.09625, 1.0862500000000002, 1.0806250000000002, 1.088125, 1.07375, 1.09, 1.08875, 1.0818750000000001, 1.0775000000000003, 1.086875, 1.0843749999999999, 1.0887499999999999, 1.0825, 1.08625, 1.084375, 1.0831250000000001, 1.075625, 1.08125, 1.090625, 1.08125, 1.089375, 1.0850000000000002, 1.0750000000000002, 1.084375, 1.0831250000000001, 1.0812500000000003, 1.09, 1.0662500000000001, 1.0637500000000002, 1.080625, 1.0718750000000001, 1.073125, 1.086875, 1.0743750000000003, 1.0868750000000003, 1.073125, 1.0750000000000002, 1.075625, 1.074375, 1.07125, 1.085, 1.075625, 1.0825, 1.073125, 1.069375, 1.0806250000000002, 1.0675, 1.074375, 1.0818750000000001, 1.07125, 1.0706250000000002, 1.0737500000000002, 1.06625, 1.07625, 1.0662500000000001, 1.073125, 1.0618750000000001, 1.0706250000000002, 1.0743750000000003, 1.0699999999999998, 1.0593750000000002, 1.0725000000000002, 1.0700000000000003, 1.074375, 1.06875, 1.07, 1.074375, 1.056875, 1.0687499999999999, 1.0768749999999998, 1.0737500000000002, 1.0612499999999998, 1.07, 1.063125, 1.073125, 1.06, 1.0818750000000001, 1.0725, 1.0787500000000003, 1.0656250000000003, 1.0768750000000002, 1.0650000000000002, 1.05375, 1.064375, 1.058125, 1.071875, 1.0693750000000002, 1.0681250000000002, 1.0575, 1.0625000000000002, 1.0637500000000002, 1.0612500000000002, 1.066875, 1.0575, 1.0668749999999998, 1.0581250000000002, 1.06, 1.0631250000000003, 1.048125, 1.060625, 1.05875, 1.065625, 1.065, 1.0600000000000003, 1.05125, 1.056875, 1.06625, 1.065, 1.055625, 1.0656249999999998, 1.0681250000000002, 1.0575, 1.0625, 1.0531249999999999, 1.0525, 1.0525, 1.0462500000000001, 1.05875, 1.0618750000000001, 1.0587500000000003, 1.0581250000000002, 1.06375, 1.05375, 1.0537500000000002, 1.058125, 1.0618750000000001, 1.0643749999999998, 1.055625, 1.0518750000000001, 1.0431249999999999, 1.0493750000000002, 1.0518750000000001, 1.0525, 1.06125, 1.05, 1.0543750000000003, 1.0550000000000002, 1.058125, 1.053125, 1.0475, 1.053125, 1.0506250000000004, 1.0487500000000003, 1.0556250000000003, 1.0412499999999998, 1.046875, 1.054375, 1.050625, 1.055625, 1.053125, 1.0525000000000002, 1.05375, 1.0525, 1.055625, 1.04875, 1.0475, 1.0406250000000001, 1.0637500000000002, 1.054375, 1.0431249999999999, 1.055, 1.041875, 1.043125, 1.048125, 1.0537500000000002, 1.0506250000000001, 1.041875, 1.045625, 1.0506250000000004, 1.0425, 1.0506250000000001, 1.04875, 1.0475, 1.04375, 1.0456249999999998, 1.045625, 1.043125, 1.04625, 1.03625, 1.0499999999999998, 1.0493750000000004, 1.0400000000000003, 1.0481250000000002, 1.035625, 1.055625, 1.055, 1.04125, 1.0399999999999998, 1.0387499999999998, 1.0462500000000001, 1.03875, 1.045, 1.04125, 1.04125, 1.0418749999999999, 1.04125, 1.0431249999999999, 1.035625, 1.0399999999999998, 1.04125, 1.0375, 1.04, 1.03625, 1.041875, 1.04375, 1.043125, 1.0437500000000002, 1.0475, 1.039375, 1.039375, 1.0399999999999998, 1.035, 1.0356249999999998, 1.0349999999999997, 1.035625, 1.039375, 1.04, 1.04, 1.024375, 1.045, 1.036875, 1.036875, 1.038125, 1.03125, 1.035, 1.044375, 1.030625, 1.034375, 1.0425, 1.03375, 1.0325, 1.038125, 1.03125, 1.031875, 1.03875, 1.036875, 1.02375, 1.035625, 1.03, 1.031875, 1.03625, 1.043125, 1.03875, 1.041875, 1.039375, 1.026875, 1.0325000000000002, 1.030625, 1.024375, 1.0374999999999999, 1.035, 1.0275, 1.035, 1.0287499999999998, 1.029375, 1.0318749999999999, 1.03375, 1.0362500000000001, 1.0356249999999998, 1.0299999999999998, 1.044375, 1.0325, 1.03, 1.0318749999999999, 1.02875, 1.033125, 1.03625, 1.034375, 1.0331249999999998, 1.0274999999999999, 1.0274999999999999, 1.0224999999999997, 1.030625, 1.0324999999999998, 1.0249999999999997, 1.030625, 1.03125, 1.025, 1.0256249999999998, 1.02625, 1.035, 1.0231249999999998, 1.028125, 1.0262499999999999, 1.0299999999999998, 1.03125, 1.0262499999999999, 1.03, 1.0399999999999998, 1.0318749999999999, 1.0293749999999997, 1.0362500000000001, 1.02375, 1.04, 1.0325, 1.0318750000000003, 1.026875, 1.0343749999999998, 1.034375, 1.025, 1.02125, 1.0318750000000003, 1.0299999999999998, 1.0262499999999999, 1.0237499999999997, 1.024375, 1.026875, 1.025, 1.028125, 1.0281249999999997, 1.03, 1.025, 1.0262499999999999, 1.0262499999999999, 1.0231249999999998, 1.02875, 1.0275, 1.038125, 1.0231249999999998, 1.0281249999999997, 1.025625, 1.024375, 1.028125, 1.02125, 1.026875, 1.0231249999999998, 1.0224999999999997]


**Define The General Exponential Function to Fit on The Experimental Data**


```python
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
```

**Plot the Number of Non-observed Values With Respect to the Number of Available Ciphertexts - Overview**


```python
x_start_point = 1
x_end_point = max_number_of_queries
x_data = range(x_start_point, x_end_point)

y_start_point = 0
y_end_point = 256

cmap = plt.get_cmap('hsv')
colors = [cmap(i) for i in np.linspace(0, 1, 17)]
for i in range(len(number_of_faults)):
    y_data = candidates[i][x_start_point - 1:]
    plt.plot(x_data, y_data,\
             color=colors[i], label='$\lambda = %d$' % (i + 1), linewidth=0.6)
    m = 2**8 - (i + 1)
    expect_number_of_queries = np.ceil((m*harmonic_number(m)))
    plt.plot([expect_number_of_queries]*2, [y_start_point, y_end_point],\
                '--', color=colors[i], label='', linewidth=0.6)

plt.legend(fontsize='xx-small', ncol=1, loc='best')

x_tick_step = 215
y_tick_step = 16
plt.xticks(list(range(0, max_number_of_queries, x_tick_step)))
plt.yticks([1] + list(range(16, 260, y_tick_step)))
# plt.xlim(-100, 2000)
# plt.ylim(0, 260)
plt.grid(True)
plt.xlabel('$N$: Number of known ciphertexts')
plt.ylabel('Number of non-observed values')
# plt.show()
plt.savefig("overview_diagram_of_non_observed_values.svg", format='svg', dpi=1200)
```


    
![png](output_29_0.png)
    


**Plot the Number of Non-observed Values With Respect to The Number of Available Ciphertexts - Close up**


```python
x_start_point = 1
x_end_point = max_number_of_queries
x_data = range(x_start_point, x_end_point)

y_start_point = 0
y_end_point = 256

cmap = plt.get_cmap('hsv')
colors = [cmap(i) for i in np.linspace(0, 1, 17)]
for i in range(len(number_of_faults)):
    y_data = candidates[i][x_start_point - 1:]
    plt.plot(x_data, y_data,\
             color=colors[i], label='$\lambda = %d$' % (i + 1), linewidth=0.6)
    m = 2**8 - (i + 1)
    expect_number_of_queries = np.ceil((m*harmonic_number(m)))
    plt.plot([expect_number_of_queries]*2, [y_start_point, y_end_point],\
                '--', color=colors[i], label='', linewidth=0.6)

plt.legend(fontsize='xx-small', ncol=2, loc='best')

x_tick_step = 150
y_tick_step = 1
plt.xticks(list(range(700, 2000, x_tick_step)))
plt.yticks(list(range(1, 20, y_tick_step)))
plt.xlim(600, 2000)
plt.ylim(0, 20)
plt.grid(True)
plt.xlabel('$N$: Number of known ciphertexts')
plt.ylabel('Number of non-observed values')
# plt.show()
plt.savefig("close_up_diagram_of_non_observed_values.svg", format='svg', dpi=1200)
```


    
![png](output_31_0.png)
    


**Fitt an Exponential Curve on the Derived Data - Overview**


```python
import warnings
warnings.filterwarnings('ignore')

x_start_point = 1
x_end_point = max_number_of_queries
x_data = np.arange(x_start_point, x_end_point)

y_start_point = 0
y_end_point = 256

cmap = plt.get_cmap('hsv')
colors = [cmap(i) for i in np.linspace(0, 1, 17)]
for i in range(len(number_of_faults)):
    y_data = candidates[i][x_start_point - 1:]
    # Fitt a curve to data
    popt, pcov = curve_fit(func, x_data, y_data)
    plt.plot(x_data, func(x_data, *popt),\
             color=colors[i],
             label='$\lambda = %d, a=%5.3f, b=%5.3f, c=%5.3f$' % (i + 1, *popt), linewidth=0.6)
    m = 2**8 - (i + 1)
    expect_number_of_queries = np.ceil((m*harmonic_number(m)))
    plt.plot([expect_number_of_queries]*2, [y_start_point, y_end_point],\
                '--', color=colors[i], label='', linewidth=0.6)

plt.legend(fontsize='xx-small', ncol=2, loc='best')
plt.title("$y = a \cdot e^{-b \cdot N} + c$")

x_tick_step = 215
y_tick_step = 16
plt.xticks(list(range(0, max_number_of_queries, x_tick_step)))
plt.yticks([1] + list(range(16, 260, y_tick_step)))
# plt.xlim(-100, 2000)
# plt.ylim(0, 260)
plt.grid(True)
plt.xlabel('$N$: Number of known ciphertexts')
plt.ylabel("$y$")
# plt.show()
plt.savefig("overview_fit_on_non_observed_values.svg", format='svg', dpi=1200)
```


    
![png](output_33_0.png)
    


**Fitt an Exponential Curve on The Derived Data - Close up**


```python
x_start_point = 1
x_end_point = max_number_of_queries
x_data = np.arange(x_start_point, x_end_point)

y_start_point = 0
y_end_point = 260

cmap = plt.get_cmap('hsv')
colors = [cmap(i) for i in np.linspace(0, 1, 17)]
for i in range(len(number_of_faults)):
    y_data = candidates[i][x_start_point - 1:]
    # Fitt a curve to data
    popt, pcov = curve_fit(func, x_data, y_data)
    plt.plot(x_data, func(x_data, *popt),\
             color=colors[i],
             label='$\lambda = %d, a=%5.3f, b=%5.3f, c=%5.3f$' % (i + 1, *popt), linewidth=0.9)
    # Draw a vertical line to show the expected number of queries based on our estimation
    m = 2**8 - (i + 1)
    expect_number_of_queries = np.ceil((m*harmonic_number(m)))
    plt.plot([expect_number_of_queries]*2, [y_start_point, y_end_point],\
                '--', color=colors[i], label='', linewidth=0.6)

plt.legend(fontsize='xx-small', ncol=1, loc='best')
plt.title("$y = a \cdot e^{-b \cdot N} + c$")

x_tick_step = 150
y_tick_step = 1
plt.xticks(list(range(700, 2000, x_tick_step)))
plt.yticks([1] + list(range(1, 20, y_tick_step)))
# plt.margins(x=0.05, y=0.05)
plt.xlim(600, 2000)
plt.ylim(0, 20)
plt.grid(True)
plt.xlabel('$N$: Number of known ciphertexts')
plt.ylabel('$y$')
# plt.legend(bbox_to_anchor=(-0.11, 0.5), loc='center right', fontsize='xx-small')
# plt.show()
plt.savefig("close_up_fit_on_non_obsereved_values.svg", format='svg', dpi=1200)
```


    
![png](output_35_0.png)
    


## Experiment 2 - Implement Algorithm 1 to Find deltaj = skR[0] + skR[j]
In this experiment we aim to check how Algorithm 1 in our paper works in practice.


```python
def alg1_to_find_delta(D0, Dj, number_of_faults):
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
    plt.bar(keys, values)
    # print("Number of faults: %d" % number_of_faults)
    # print("Number of appearance for each delta:\n%s" % delta_counters)
    candidates = [delta for delta in delta_counters.keys() if delta_counters[delta] == number_of_faults]
    return candidates
```

**Run Faulty AES for Sufficiently Large Number of Random Plaintexts And Collect The Observed As Well As Non-observed Values for Each Output Byte**


```python
def generate_deltas_for_large_number_of_ciphertexts(number_of_faults=2):
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
    #print("Expected number of queries: %d, bias: %d" % (expect_number_of_queries, bias))
    D = [[] for _ in range(16)]
    for col in range(4):
        for row in range(4):
            j = 4*col + row
            D[j] = non_observed_bytes[col][row]
            # print('D_{%d}: %s' % (j, D[j]))
    return number_of_random_plaintexts, D, faulty_aes.dictionary_of_replacement, last_round_key
```


```python
nf = 4
position = 5
number_of_ciphertexts, DD, true_fault_mapping, true_last_round_key =\
     generate_deltas_for_large_number_of_ciphertexts(number_of_faults=nf)
delta_candidates = alg1_to_find_delta(DD[0], DD[position], number_of_faults=nf)
print("Number of available ciphertexts: %d" % number_of_ciphertexts)
print("Candidates for delta%d: %s" % (position, delta_candidates))
print("skR[0] xor skR[%d]: %d" % (position, true_last_round_key[0] ^ true_last_round_key[position]))
```

    Number of available ciphertexts: 3080
    Candidates for delta5: [223]
    skR[0] xor skR[5]: 223



    
![png](output_40_1.png)
    


## Experiment 3 - Implement Algorithm 2 to Find deltaj = skR[0] + skR[j]
In this experiment we aim to implement the algorithm 2 to see how it works in practice.

**Implement Algorithm 2 to Find deltaj = skR0 + skRj When a Limited Number of Known Ciphertexts is Available**


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


```python
def generate_deltas(number_of_faults=2, number_of_non_observed_values=3):
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
    #print("Expected number of queries: %d, bias: %d" % (expect_number_of_queries, bias))
    D = [[] for _ in range(16)]
    for col in range(4):
        for row in range(4):
            j = 4*col + row
            D[j] = non_observed_bytes[col][row]
            # print('D_{%d}: %s' % (j, D[j]))
    return number_of_random_plaintexts, D, faulty_aes.dictionary_of_replacement, last_round_key
```


```python
def compute_lamda_prime_from_lambda_and_N(N, lam):
    a = 2**8 - lam
    b = 1.0/(2**8 - lam)
    c = lam
    output = a*exp(-b*N) + c
    return output
lam = 4
m = 2**8 - lam
compute_lamda_prime_from_lambda_and_N(int(m*harmonic_number(m)) - 100, 3)
```




    3.856965560894711




```python
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
    print("Experiment no: %3d" % i)
print("Number of known ciphertexts: %d" % number_of_known_ciphertexts)
print("Average number of candidates for deltaj in each output byte: %0.2f" % mean(number_of_candidates))
mean_num_of_candidates = mean(number_of_candidates)
```

    Experiment no:   0
    Experiment no:   1
    Experiment no:   2
    Experiment no:   3
    Experiment no:   4
    Experiment no:   5
    Experiment no:   6
    Experiment no:   7
    Experiment no:   8
    Experiment no:   9
    Experiment no:  10
    Experiment no:  11
    Experiment no:  12
    Experiment no:  13
    Experiment no:  14
    Experiment no:  15
    Experiment no:  16
    Experiment no:  17
    Experiment no:  18
    Experiment no:  19
    Experiment no:  20
    Experiment no:  21
    Experiment no:  22
    Experiment no:  23
    Experiment no:  24
    Experiment no:  25
    Experiment no:  26
    Experiment no:  27
    Experiment no:  28
    Experiment no:  29
    Experiment no:  30
    Experiment no:  31
    Experiment no:  32
    Experiment no:  33
    Experiment no:  34
    Experiment no:  35
    Experiment no:  36
    Experiment no:  37
    Experiment no:  38
    Experiment no:  39
    Experiment no:  40
    Experiment no:  41
    Experiment no:  42
    Experiment no:  43
    Experiment no:  44
    Experiment no:  45
    Experiment no:  46
    Experiment no:  47
    Experiment no:  48
    Experiment no:  49
    Experiment no:  50
    Experiment no:  51
    Experiment no:  52
    Experiment no:  53
    Experiment no:  54
    Experiment no:  55
    Experiment no:  56
    Experiment no:  57
    Experiment no:  58
    Experiment no:  59
    Experiment no:  60
    Experiment no:  61
    Experiment no:  62
    Experiment no:  63
    Experiment no:  64
    Experiment no:  65
    Experiment no:  66
    Experiment no:  67
    Experiment no:  68
    Experiment no:  69
    Experiment no:  70
    Experiment no:  71
    Experiment no:  72
    Experiment no:  73
    Experiment no:  74
    Experiment no:  75
    Experiment no:  76
    Experiment no:  77
    Experiment no:  78
    Experiment no:  79
    Experiment no:  80
    Experiment no:  81
    Experiment no:  82
    Experiment no:  83
    Experiment no:  84
    Experiment no:  85
    Experiment no:  86
    Experiment no:  87
    Experiment no:  88
    Experiment no:  89
    Experiment no:  90
    Experiment no:  91
    Experiment no:  92
    Experiment no:  93
    Experiment no:  94
    Experiment no:  95
    Experiment no:  96
    Experiment no:  97
    Experiment no:  98
    Experiment no:  99
    Number of known ciphertexts: 1173
    Average number of candidates for deltaj in each output byte: 2.15



```python
a = log2((mean_num_of_candidates**15)*256)
print("Number of key candidates: 2^(%0.02f)" % a)
```

    Number of key candidates: 2^(24.55)

## Algorithm 3 to Find D0*

Our codes concerning Algorithm 3 are located in the following path:

  - Markdown: [KeyRecovery2.md](KeyRecovery2.md)
  - Jupyter: [KeyRecovery2.ipynb](KeyRecovery2.ipynb)
  
## Algorithm 4 - Key Recovery 1

Our codes concerning Algorithm 4 are located in the following path:

  - Markdown: [KeyRecovery1.md](KeyRecovery1.md)
  - Jupyter: [KeyRecovery1.ipynb](KeyRecovery1.ipynb)

## Parallel Key Recovery 1

Our codes to paralleize algorithm 4 are located in the following path:

  - Markdown: [ParallelKeyRecovery.md](ParallelKeyRecovery.md)
  - Jupyter: [ParallelKeyRecovery.ipynb](ParallelKeyRecovery.ipynb)
  - Python scripty: [ParallelKeyRecovery.py](ParallelKeyRecovery.py):
    - To run this script, issue this command: `python3 ParallelKeyRecovery.py`

## Algorithm 4 - Key Recovery 1 For Faulty S-boxes Derived in the Laboratory

Our codes concerning Algorithm 4 for some faulty S-boxes derived in the laboratory:

  -[KeyRecovery1ForGivenFaultySboxesFromTheLab.md](KeyRecovery1ForGivenFaultySboxesFromTheLab.md)
  -[KeyRecovery1ForGivenFaultySboxesFromTheLab.ipynb](KeyRecovery1ForGivenFaultySboxesFromTheLab.ipynb)


## Algorithm 5 - Key Recovery 2

Our codes implementing the Algorithm 5 can be found in the following path:

  - Markdown: [KeyRecovery2.md](KeyRecovery2.md)
  - Jupyter: [KeyRecovery2.ipynb](KeyRecovery2.ipynb)

## Algorithm 5 - Key recovery 2 For Faulty S-boxes Derived in the Laboratory

Our codes concerning Algorithm 5 for some faulty S-boxes derived in the laboratory:

  - Markdown: [KeyRecovery2ForGivenSboxesFromTheLab.md](KeyRecovery2ForGivenSboxesFromTheLab.md)
  - Jupyter: [KeyRecovery2ForGivenSboxesFromTheLab.ipynb](KeyRecovery2ForGivenSboxesFromTheLab.ipynb)

## License

```sh
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