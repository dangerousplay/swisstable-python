"""Simple example for embedding SSE2 assembly in Cython projects.

The purpose of this project is to provide documentation, since this information was a bit hard to find.

Library module.
"""

from __future__ import division, print_function, absolute_import

ctypedef int bitmask


cdef int _trailing_zeros(int n):
    """
    :param n: The input integer for which trailing zeros are to be counted.
    :return: The number of trailing zeros in the binary representation of the input integer.
    """
    if n == 0:
        return 16

    cdef int count = 0

    if (n & 0xFF) == 0:
        n >>= 8
        count += 8

    if (n & 0xF) == 0:
        n >>= 4
        count += 4

    if (n & 0x3) == 0:
        n >>= 2
        count += 2

    if (n & 0x1) == 0:
        count += 1

    return count

# From https://github.com/Technologicat/cython-sse-example/tree/master
# http://stackoverflow.com/questions/11228855/header-files-for-x86-simd-intrinsics
#
# MMX:     mmintrin.h
# SSE:    xmmintrin.h
# SSE2:   emmintrin.h
# SSE3:   pmmintrin.h
# SSSE3:  tmmintrin.h
# SSE4.1: smmintrin.h
# SSE4.2: nmmintrin.h
# SSE4A:  ammintrin.h
# AES:    wmmintrin.h
# AVX:    immintrin.h
# AVX512: zmmintrin.h
#
cdef extern from "emmintrin.h":
    # Two things happen here:
    # - this definition tells Cython that, at the abstraction level of the Cython language, __m128d "behaves like a double" and __m128i "behaves like a long array [2]"
    # - at the C level, the "cdef extern from" (above) makes the generated C code look up the exact definition from the original header
    #
    ctypedef double __m128d
    ctypedef long long int[2] __m128i

    # Declare any needed extern functions here; consult $(locate emmintrin.h) and SSE assembly documentation.
    #
    # For example, to pack an (unaligned) double pair, to perform addition and multiplication (on packed pairs),
    # and to unpack the result, one would need the following:
    #
    __m128i _mm_set_epi8(char __q15, char __q14, char __q13, char __q12,
                 char __q11, char __q10, char __q09, char __q08,
                 char __q07, char __q06, char __q05, char __q04,
                 char __q03, char __q02, char __q01, char __q00) nogil

    __m128i _mm_store_si128(__m128i * dest, __m128i a) nogil
    __m128i _mm_set1_epi8(char v) nogil
    __m128i _mm_cmpeq_epi8(__m128i __A, __m128i __B) nogil
    int _mm_movemask_epi8(__m128i __A) nogil


cdef bitmask _find_matches(char hash, char[16] keys):
    """
    :param hash: a byte representing the hash value to compare against
    :param keys: an array of 16 characters representing the keys to check for a match
    :return: an integer representing the bitmask of matches found

    This method takes a hash value and an array of control keys, and returns an integer bitmask of matches found.
    The method compares the hash value to each element in the keys array and checks for equality.
    If a match is found, the corresponding bit in the bitmask is set to 1, otherwise it is set to 0.
    The method uses SIMD (Single Instruction, Multiple Data) operations to efficiently perform the comparison and bitmask operations.
    
    _mm_set1_epi8(hash): This intrinsic sets each byte in an m128i value to the specific byte pattern. 
                         It replicates hash 16 times to create a 128-bit value.
    
    _mm_set_epi8(keys[15], ..., keys[0]): This intrinsic sets the 16 signed 8-bit integer values in reverse order.
    
    _mm_cmpeq_epi8(...): This macro compares for equality each of the corresponding 16-byte items from the two 128-bit values provided as parameters.
    
    _mm_movemask_epi8(...): This operation takes a 128-bit value and moves the most significant bit of each byte into a 16-bit integer.
                            This is how the resulting bitmask is formed. 
                            This integer will be returned as the result of _find_matches method.
    
    """
    return _mm_movemask_epi8(
        _mm_cmpeq_epi8(
            _mm_set1_epi8(hash),
            _mm_set_epi8(
                keys[15], keys[14], keys[13], keys[12],
                keys[11], keys[10], keys[9],  keys[8],
                keys[7],  keys[6],  keys[5],  keys[4],
                keys[3],  keys[2],  keys[1],  keys[0]
            )
        )
    )


def find_matches(int hash, bytes keys) -> bitmask:
    """
    :param hash: The hash value to search for matches.
    :param keys: The list of keys to search for matches in.
    :return: The matches found for the given hash and keys.

    This method takes a hash value and a list of control keys and searches for matches among the keys.
    It returns the matches found for the given hash and keys.

    Note that the hash value and the keys should be of appropriate type - hash should be an integer and keys should be a bytes object.

    Example usage:
        hash = ord('a')
        keys = ['T', 'h', 'i', 's', 'i', 's', 'a', 't', 'e', 's', 't', 'z', '1', '2', '3', '0']
        keys = list(map(ord, keys))
        matches = find_matches(hash, keys)
    """

    cdef char c_hash = <char>hash
    cdef char[16] c_keys
    cdef int keys_size = len(keys)

    if keys_size < 16:
        raise ValueError(f"Invalid keys size: {len(keys)}")

    for i in range(16):
        c_keys[i] = <char> keys[i]

    return _find_matches(c_hash, c_keys)

def trailing_zeros(long number):
    """
    :param number: The input number
    :return: The number of trailing zeros in the binary representation of the input number
    """
    return _trailing_zeros(number)

def print_bitmask(bitmask match):
    """
    Prints the bits of a given bitmask.

    :param match: The bitmask to be printed.
    :return: None
    """

    cdef char * results = <char *> &match
    cdef char value

    for i in range(2):
        value = results[i]

        for j in range(8):
            print((value >> (7 - j)) & 1, end='')

        print("")


def next_match(bitmask bitset):
    cdef int next = _trailing_zeros(bitset)

    # Remove the match found from the bitset
    bitset &= ~(1 << next)

    return next, bitset
