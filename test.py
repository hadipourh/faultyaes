"""
This module checks the correctness of AES implementation. To do so, 
we compare a sample plaintext/ciphertext generated by our implementation 
with the test vector given in NIST reference document: 
https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/AES_Core128.pdf
"""

from faultyaes import *

master_key = 0x2B7E151628AED2A6ABF7158809CF4F3C
plaintext = 0x6BC1BEE22E409F96E93D7E117393172A
expected_ciphertext = 0x3AD77BB40D7A3660A89ECAF32466EF97


def test_encryption():
    faulty_aes = AES(master_key)
    faulty_aes.apply_fault(number_of_faults=0)
    derived_ciphertext = faulty_aes.encrypt(plaintext)
    if derived_ciphertext == expected_ciphertext:
        assert True
    else:
        assert False, "AES encryption doesn't work correctly :-("

def test_decryption():
    faulty_aes = AES(master_key)
    faulty_aes.apply_fault(number_of_faults=0)
    derived_ciphertext = faulty_aes.encrypt(plaintext)
    derived_plaintext = faulty_aes.decrypt(derived_ciphertext)
    if derived_plaintext == plaintext:        
        assert True
    else:
        assert False, "AES decryption doesn't work correctly :-("
