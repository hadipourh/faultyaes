# Multiple Persisten Faults Attack - Key Recovery Based on V and V* - For Given Faulty Sboxes Derived in The Laboratory

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

## Required Packages


```python
import numpy as np
from statistics import mean, variance, stdev
import random
import itertools
from fractions import Fraction
import time
from multiprocessing import Pool
import pickle
```

## Initialize Random Generator
`random.seed(a=None, version=2)`

If a is omitted or None, the current system time is used. If randomness sources are provided by the operating system, they are used instead of the system time (see the os.urandom() function for details on availability).

`random.sample(population, k, *, counts=None)`

Returns a k length list of unique elements chosen from the population sequence or set. Used for random sampling without replacement.

## Define Harmonic Number


```python
harmonic_number = lambda n: float(sum(Fraction(1, d) for d in range(1, n+1)))
```

## Parse Faulty Sboxes Derived in Laboratory


```python
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
```


```python
for i in range(256):
    if true_sb[i] != sb_look_up_6bytes_faults[i]:
        print("%3d ==> %3d" % (true_sb[i], sb_look_up_6bytes_faults[i]))
```

    215 ==> 223
    171 ==> 255
    118 ==> 255
     89 ==> 255
     71 ==> 255
    240 ==> 255


## Implement (Faulty) AES-128

* In the following implementation it is supposed that the S-boxes used in key-schedule are not faulty.


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

    def apply_fault_lab(self, number_of_faults):
        if number_of_faults == 2:
            self.faulty_sbox = sb_look_up_2bytes_faults
        elif number_of_faults == 4:
            self.faulty_sbox = sb_look_up_4bytes_faults
        elif number_of_faults == 6:
            self.faulty_sbox = sb_look_up_6bytes_faults
        elif number_of_faults == 8 or number_of_faults == 7:
            self.faulty_sbox = sb_look_up_8bytes_faults
        else:
            return "Number of faults is assumed to be in [2, 4, 6, 8] in this simulation!"
        self.dictionary_of_replacement = dict()
        for i in range(256):
            if self.sbox[i] != self.faulty_sbox[i]:
                self.dictionary_of_replacement[self.sbox[i]] = self.faulty_sbox[i]
        self.inv_faulty_sbox = [0]*(len(self.sbox))
        for i in range(len(self.faulty_sbox)):
            self.inv_faulty_sbox[self.faulty_sbox[i]] = i
            
    def apply_fault(self, number_of_faults, fault_mapping=None):
        self.faulty_sbox = self.gen_random_faulty_sbox(number_of_faults, fault_mapping=fault_mapping)
        self.inv_faulty_sbox = [0]*(len(self.sbox))
        for i in range(len(self.faulty_sbox)):
            self.inv_faulty_sbox[self.faulty_sbox[i]] = i
    
    def gen_random_faulty_sbox(self, number_of_faults, fault_mapping=None):
        sbox_values_set = set(self.sbox)
        if fault_mapping == None:            
            random_positions = random.sample(range(len(self.sbox)), number_of_faults)
            original_values = [self.sbox[i] for i in random_positions]
            original_values_complement = list(sbox_values_set.difference(set(original_values)))
            faulty_values = random.sample(original_values_complement, number_of_faults)
            self.dictionary_of_replacement = dict(zip(original_values, faulty_values))
        else:
            self.dictionary_of_replacement = fault_mapping
            random_positions = [i for i in range(256) if self.sbox[i] in self.dictionary_of_replacement.keys()]            
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

        return self.round_keys[40:44]
    
    def derive_round_keys_from_last_round_key(self, last_rkey):        
        for i in range(40, 44):
            self.round_keys[i] = last_rkey[i - 40]
        for i in range(39, -1, -1):
            if i % 4 == 0:
                temp = []
                byte = self.round_keys[i + 4][0] \
                     ^ self.sbox[self.round_keys[i + 3][1]] \
                     ^ self.Rcon[i // 4 + 1]
                temp.append(byte)

                for j in range(1, 4):
                    byte = self.round_keys[i + 4][j] \
                         ^ self.sbox[self.round_keys[i + 3][(j + 1) % 4]]
                    temp.append(byte)
                self.round_keys[i] = temp
            else:
                temp = []
                for j in range(4):
                    byte = self.round_keys[i + 4][j] \
                         ^ self.round_keys[i + 3][j]
                    temp.append(byte)
                self.round_keys[i] = temp    

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

    def decrypt_and_count(self, ciphertext, Vi, Vi_star): 
        Vi = set(Vi)
        Vi_star = set(Vi_star)
        self.cipher_state = text2matrix(ciphertext)
        self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.__inv_shift_rows(self.cipher_state)
        yR = [[self.cipher_state[col][row] for row in range(4)] for col in range(4)]
        for col in range(4):
            yR = [[self.cipher_state[col][row] for row in range(4)] for col in range(4)]
            yRcol = set(yR[col])
            if yRcol.intersection(Vi_star)  == set():
                self.__inv_sub_bytes(yR)
                self.__add_round_key(yR, self.round_keys[4*9 : 4*(9 + 1)])
                self.__inv_mix_columns(yR)
                yR_1Col = set(yR[col])
                if yR_1Col.intersection(Vi) != set():
                    return 0
        return 1

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

## Find deltaj = skR0 + skRj Assuming that Enough Number of Known Ciphertexts Are Available


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
    output = [d_star for d_star in output if output_temp[d_star] >= 8]
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
    #faulty_aes.apply_fault(number_of_faults)
    faulty_aes.apply_fault_lab(number_of_faults=number_of_faults)
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

## Check the Quality of Our Algorithm for Dtecting D0*


```python
nf = 6
m = 2**8 - nf
number_of_given_ciphertexts = int(np.ceil(2*m*harmonic_number(m)))
known_ciphertexts, k_v_candidates, last_round_key, fault_mapping, D, D_star\
                 = generate_input_data_for_key_recovery(number_of_faults=nf, number_of_known_ciphertexts=number_of_given_ciphertexts)
D0_star = find_D0_star(number_of_faults=nf, D=D, D_star_candidates=D_star, last_round_key=last_round_key)
print(D0_star)
```

    Number of delta candidates: 1
    candidate: 38, counter: 15
    candidate: 6, counter: 11
    [38, 6]



```python
[k ^ last_round_key[0] for k in D0_star]
```




    [255, 223]



## Provide Required Data for Key Recovery


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
        # aes_instance.apply_fault(number_of_faults=number_of_faults, fault_mapping=fault_mapping)
        aes_instance.apply_fault_lab(number_of_faults=number_of_faults)
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
                k_v_candidates[Ki][2] = aes_instance.decrypt_and_count(this_cipher, k_v_candidates[Ki][0], k_v_candidates[Ki][1])
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

## Key Recovery


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
    Time used by key recovery: 0.12 Seconds
    ------------- KEY RECOVERY WAS FINISHED -------------
    [1]
    {(124, 15, 196, 122, 76, 41, 43, 88, 157, 120, 52, 154, 181, 166, 185, 216): [(124, 15, 196, 122, 76, 41, 43, 88, 157, 120, 52, 154, 181, 166, 185, 216)]}

