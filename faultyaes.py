"""
Author: Hosein Hadipour
Date: Jan 27, 2022
I borrowed the AES-128 Python implementation from https://github.com/bozhu/AES-Python
and then modified it to an implementation simulating the multiple persistent fault injection.

Moreover, it is supposed that key schedule is not affected by the faults.
"""

import random
from fractions import Fraction


def harmonic_number(n):
    """
    Compute the n'th harmonic number
    
    :param int n: a natural number
    :return: n'th harmonic number
    :rtype: float
    """

    output = sum(Fraction(1, d) for d in range(1, n+1))
    output = float(output)    
    return output

xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)

def text2matrix(text):
    """
    Represent an integer as a 4 by 4 array of bytes

    :param int text: an arbitrary integer number
    :return: a 4 by 4 array of bytes
    :rtype: int[4][4]
    """

    matrix = []
    for i in range(16):
        byte = (text >> (8 * (15 - i))) & 0xFF
        if i % 4 == 0:
            matrix.append([byte])
        else:
            matrix[i // 4].append(byte)
    return matrix

def matrix2text(matrix):
    """
    Convert a 4 by 4 array of bytes into an integer

    :param int[4][4] matrix: a 4 by 4 array of bytes (AES state array)
    :return: an integer representation of the AES state array
    :rtype: int
    """

    text = 0
    for i in range(4):
        for j in range(4):
            text |= (matrix[i][j] << (120 - 8 * (4 * i + j)))
    return text

class AES:
    class_cnt = 0
    def __init__(self, master_key):
        """
        Initialize the S-box, round constants and the round keys

        :param int master_key: master key of AES-128
        """

        AES.class_cnt += 1        
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
    
    def apply_fault(self, number_of_faults, fault_mapping=None):
        """
        Call gen_random_faulty_sbox to generate a random
        faulty S-box and initialize its corresponding inverse

        :param int number_of_faults: number of faults on the reference S-box lookup table
        :param dict fault_mapping: mapping of fault injection
        """

        self.faulty_sbox = self.gen_random_faulty_sbox(number_of_faults, fault_mapping=fault_mapping)
        self.inv_faulty_sbox = [0]*(len(self.sbox))
        for i in range(len(self.faulty_sbox)):
            self.inv_faulty_sbox[self.faulty_sbox[i]] = i
    
    def gen_random_faulty_sbox(self, number_of_faults, fault_mapping=None):
        """
        Generate a random faulty S-box with a certain number of faults

        :param int number_of_faults:
        """

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

    def apply_fault_lab(self, labfaultysbox):
        """
        Set the faulty S-box to the faulty S-box derived from the laboratory.

        :param int[256] labfaultysbox: faulty S-box derived from laboratory.
        """

        self.faulty_sbox = labfaultysbox
        self.dictionary_of_replacement = dict()
        for i in range(256):
            if self.sbox[i] != self.faulty_sbox[i]:
                self.dictionary_of_replacement[self.sbox[i]] = self.faulty_sbox[i]
        self.inv_faulty_sbox = [0]*(len(self.sbox))
        for i in range(len(self.faulty_sbox)):
            self.inv_faulty_sbox[self.faulty_sbox[i]] = i

    def change_key(self, master_key):
        """
        Reinitialize round keys

        :param int master_key: new master key of AES
        :return: the last round key
        :rtype: array of ints
        """

        self.round_keys = text2matrix(master_key)

        for i in range(4, 4 * 11):
            self.round_keys.append([])
            if i % 4 == 0:
                byte = self.round_keys[i - 4][0]\
                     ^ self.sbox[self.round_keys[i - 1][1]]\
                     ^ self.Rcon[i // 4]
                self.round_keys[i].append(byte)

                for j in range(1, 4):
                    byte = self.round_keys[i - 4][j]\
                         ^ self.sbox[self.round_keys[i - 1][(j + 1) % 4]]
                    self.round_keys[i].append(byte)
            else:
                for j in range(4):
                    byte = self.round_keys[i - 4][j]\
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
        """
        Perfrom the round function 10 times to encrypt the plaintext

        :param int plaintext: plaintext
        :return: the ciphertext
        :rtype: int
        """

        self.plain_state = text2matrix(plaintext)

        self.__add_round_key(self.plain_state, self.round_keys[:4])

        for i in range(1, 10):
            self.__round_encrypt(self.plain_state, self.round_keys[4 * i : 4 * (i + 1)])

        self.__sub_bytes(self.plain_state)
        self.__shift_rows(self.plain_state)
        self.__add_round_key(self.plain_state, self.round_keys[40:])

        return matrix2text(self.plain_state)

    def decrypt(self, ciphertext):
        """
        Perform the inverse of round function 10 times to decrypt the ciphertext

        :param int ciphertext: ciphertext
        :retrun: plaintext
        :rtype: int
        """

        self.cipher_state = text2matrix(ciphertext)

        self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.__inv_shift_rows(self.cipher_state)
        self.__inv_sub_bytes(self.cipher_state)

        for i in range(9, 0, -1):
            self.__round_decrypt(self.cipher_state, self.round_keys[4 * i : 4 * (i + 1)])

        self.__add_round_key(self.cipher_state, self.round_keys[:4])

        return matrix2text(self.cipher_state)
    
    def decrypt_and_count1(self, ciphertext, Vi):        
        self.cipher_state = text2matrix(ciphertext)
        cnt = 0
        self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.__inv_shift_rows(self.cipher_state)
        for col in range(4):
            for row in range(4):
                if self.cipher_state[col][row] in Vi:
                    return cnt
        self.__inv_sub_bytes(self.cipher_state)          
        # cnt += 1
        for i in range(9, 0, -1):            
            self.__add_round_key(self.cipher_state, self.round_keys[4*i : 4*(i + 1)])
            self.__inv_mix_columns(self.cipher_state)
            self.__inv_shift_rows(self.cipher_state)
            for col in range(4):
                for row in range(4):
                    if self.cipher_state[col][row] in Vi:
                        return cnt
            cnt += 1
            self.__inv_sub_bytes(self.cipher_state)

        self.__add_round_key(self.cipher_state, self.round_keys[:4])
        return cnt

    def decrypt_and_count2(self, ciphertext, Vi, Vi_star): 
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
        """
        Perform key addition

        :param int[4][4] s: state array of AES
        :param int[4][4] k: round key
        """
        
        for i in range(4):
            for j in range(4):
                s[i][j] ^= k[i][j]


    def __round_encrypt(self, state_matrix, key_matrix):
        """
        Perfrom a full round function of AES encryption
        """

        self.__sub_bytes(state_matrix)
        self.__shift_rows(state_matrix)
        self.__mix_columns(state_matrix)
        self.__add_round_key(state_matrix, key_matrix)


    def __round_decrypt(self, state_matrix, key_matrix):
        """
        Perform a full round function of AES decryption
        """

        self.__add_round_key(state_matrix, key_matrix)
        self.__inv_mix_columns(state_matrix)
        self.__inv_shift_rows(state_matrix)
        self.__inv_sub_bytes(state_matrix)

    def __sub_bytes(self, s):
        """
        Apply S-box on 16 bytes of state array

        :param int[4][4] s: state array of AES
        """

        for i in range(4):
            for j in range(4):
                s[i][j] = self.faulty_sbox[s[i][j]]


    def __inv_sub_bytes(self, s):
        """
        Apply S-box inverse on 16 bytes of state array

        :param int[4][4] s: state array of AES
        """

        for i in range(4):
            for j in range(4):
                s[i][j] = self.inv_faulty_sbox[s[i][j]]


    def __shift_rows(self, s):
        """
        Apply shift row operation on state array

        :param int[4][4] s: state array of AES
        """

        s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]


    def __inv_shift_rows(self, s):
        """
        Apply the inverse of shift row on state array

        :param int[4][4] s: state array of AES
        """

        s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

    def __mix_single_column(self, a):
        """
        Apply the mixcolumn operation on a column of state array
        To see more details refer to Sec 4.1.2 in The Design of Rijndael

        :param int[4] a: a column of state array        
        """

        t = a[0] ^ a[1] ^ a[2] ^ a[3]
        u = a[0]
        a[0] ^= t ^ xtime(a[0] ^ a[1])
        a[1] ^= t ^ xtime(a[1] ^ a[2])
        a[2] ^= t ^ xtime(a[2] ^ a[3])
        a[3] ^= t ^ xtime(a[3] ^ u)


    def __mix_columns(self, s):
        """
        Apply the mixcolumn operation on the whole of state array

        :param int[4][4] s: state array of AES
        """

        for i in range(4):
            self.__mix_single_column(s[i])


    def __inv_mix_columns(self, s):
        """
        Apply the inverse of mixcolumn operation on a column of state array
        To see more details refer to Sec 4.1.2 in The Design of Rijndael

        :param int[4] a: a column of state array    
        """

        for i in range(4):
            u = xtime(xtime(s[i][0] ^ s[i][2]))
            v = xtime(xtime(s[i][1] ^ s[i][3]))
            s[i][0] ^= u
            s[i][1] ^= v
            s[i][2] ^= u
            s[i][3] ^= v

        self.__mix_columns(s)
