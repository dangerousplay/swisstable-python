import pyximport
import numpy

pyximport.install(setup_args={
  'include_dirs': [numpy.get_include()]
})

import sse_match


if __name__ == '__main__':
    # sse_match.run()

    x = 'a'
    a = x.__hash__()

    keys = bytes([
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1
    ])

    result = sse_match.find_matches(2, keys)
    sse_match.print_bitmask(result)
    zeros = sse_match.trailing_zeros(result)

    print(f"python result: {result}")
    print(f"Trailing zeros: {zeros}")

    chars = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']



    hash = ord('k')
    keys = bytes(list(map(ord, chars)))

    result = sse_match.find_matches(hash, keys)
    zeros = sse_match.trailing_zeros(result)

    print(f"Trailing zeros: {zeros}")

    pass
