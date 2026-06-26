import itertools
import random
import sys
from math import comb
from itertools import combinations
from typing import Generator


def count_sequences(length: int, gc_bases: int) -> int:
    """
    computes the possible number of sequences of a given length with a given GC count
    :param length:
    :param gc_bases:
    :return:
    """
    ways_to_place_gc, n_gc_orders, n_at_orders = get_seq_identifier_maxs(length, gc_bases)
    return ways_to_place_gc* n_gc_orders* n_at_orders


def get_seq_identifier_maxs(length: int, gc_bases: int) -> tuple[int, int, int]:
    """
    we can uniquely identify all sequences of n length with k GC bases with three numbers
    - index of gc location set
    - index of gc order set
    - index of at order set
    and since we have a known max number gc locations, gc orders, at orders we can
    actually bring that down to one number (to rule them all)
    todo: less wordy
    """
    ways_to_place_gc = comb(length, gc_bases)
    return ways_to_place_gc, (2 ** gc_bases), (2 ** (length - gc_bases))


def rand_iter_gc_orders(seq_len: int, gc_count: int) -> Generator[frozenset[int], None, None]:
    """
    chatGPT helped me write this this
    given the length of the sequence and the number which should be gc, yield possible sets of
    indices for gc bases in random order
    """
    # Calculate the total number of combinations
    seq_idxs = list(range(seq_len))
    total_combinations = comb(seq_len, gc_count)

    # Generate a list of indices and shuffle them
    indices = list(range(total_combinations))
    random.shuffle(indices)

    # Iterate through the shuffled indices
    for index in indices:
        # Get the corresponding combination without generating all of them
        combo = nth_combination(seq_idxs, gc_count, index)
        yield combo


def nth_combination(iterable: list[int], r: int, index: int) -> frozenset[int]:
    """
    Generate the index-th combination of iterable in lexicographic order.

    Uses the combinatorial number system: for each position, count how many
    combinations begin with each candidate element and consume `index` greedily.
    The previous implementation used index %= choose, which collapsed distinct
    indices onto the same remainder and produced duplicate combinations.
    """
    pool = tuple(iterable)
    result = []
    while r > 0:
        for k, elem in enumerate(pool):
            c = comb(len(pool) - k - 1, r - 1)
            if index < c:
                result.append(elem)
                pool = pool[k + 1:]
                r -= 1
                break
            index -= c
    return frozenset(result)


def random_number_generator(n: int) -> Generator[int, None, None]:
    """
    Lazily yield all integers from 1 to n in a uniformly random order.

    Uses an online Fisher-Yates shuffle via a virtual dict-backed array: only
    swap-modified positions are stored, so the full list is never allocated.
    This preserves early-termination efficiency for large n while fixing the
    duplicate-yield bug in the previous implementation.
    """
    virtual: dict[int, int] = {}
    for i in range(n):
        j = random.randint(i, n - 1)
        val_i = virtual.get(i, i + 1)
        val_j = virtual.get(j, j + 1)
        virtual[j] = val_i
        yield val_j


def binary_to_nucleotides(n: int, ndigits: int, opt1: str, opt2: str) -> Generator[str, None, None]:
    # Convert the number to binary, remove the '0b' prefix
    binary_str = bin(n)[2:]
    # Construct the output string with 'A' for 1 and 'T' for 0
    for _ in range(ndigits - len(binary_str)): yield opt2
    for digit in binary_str:
        yield opt1 if digit == '1' else opt2


def generate_unique_sequence(hlen: int, gc_count: int, generated_sequences: set[str] = {}) -> Generator[
    str, None, None]:
    """
    Generate a unique DNA sequence based on the specified handle length and GC count.
    Ensures that the generated sequence has not been created before and adheres to the
    specified GC count.
    """
    if gc_count > hlen:
        raise ValueError("GC count cannot be greater than the sequence length")
    if gc_count < 0 or hlen < 0:
        raise ValueError("Sequence length and GC count must be non-negative")

    # Calculate the total number of combinations
    n_gc_positionings, n_gc_orders, n_at_orders = get_seq_identifier_maxs(hlen, gc_count)

    # ok so the non(-totally)-stochastic way to do this is to iterate in a random order through an order
    # iter unique
    max_n_seqs = count_sequences(hlen, gc_count)
    for seq_uid in random_number_generator(max_n_seqs):
        assert seq_uid <= max_n_seqs

        # compute enumerated number for gc positioning
        gc_pos_id = seq_uid % n_gc_positionings  # Corrected to match n_gc_positionings
        gc_order_id = (seq_uid // n_gc_positionings) % n_gc_orders
        at_order_id = (seq_uid // (n_gc_positionings * n_gc_orders)) % n_at_orders

        assert gc_pos_id < n_gc_positionings
        assert gc_order_id < n_gc_orders
        assert at_order_id < n_at_orders

        # find set of indices of gcs
        gc_positions: frozenset = nth_combination(list(range(hlen)), gc_count, gc_pos_id)
        assert len(gc_positions) == gc_count
        # compute source strings for gcs and for ats
        gc_source = binary_to_nucleotides(gc_order_id, gc_count, "G", "C")
        at_source = binary_to_nucleotides(at_order_id, hlen - gc_count, "A", "T")
        # merge gcs and ats using gc_positions to choose at each position
        nucs = [next(gc_source)
                       if i in gc_positions else next(at_source)
                       for i in range(hlen)]
        seq = ''.join(nucs)
        if seq not in generated_sequences:
            # algorithm should not generate redundant seqs so don't have to add sequences to generated_sequences
            yield seq

    # I don't think we actually need this because it will automatically raise GeneratorExit
    raise Exception(f"Have generated all possible sequences of length {hlen} and gc content {gc_count}")


def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python script.py <handle_length> <gc_count> <size> [<filename>]")
        print("Example: python seqgen.py 8 4 100000 seq84out.txt")
        sys.exit(1)

    hlen = int(sys.argv[1])
    gc_count = int(sys.argv[2])
    size = int(sys.argv[3])

    if gc_count > hlen:
        raise ValueError("GC count too high for the given handle length.")

    print(f"Seuqnece length: {hlen}; GC count: {gc_count}")
    total_sequences = count_sequences(hlen, gc_count)
    print(f"Total sequence possibilities: {total_sequences}")

    if total_sequences < size:
        print(f"Error: requesting a sequence library size larger than the theoretical limit")
        print("Reseting the size to match the limit.")
        size = total_sequences

    print("Generating sequences...")
    seq_source = generate_unique_sequence(hlen, gc_count)
    if len(sys.argv) == 5:
        filename = sys.argv[4]

        with open(filename, "w+") as f:
            for _ in range(size):
                f.write(next(seq_source) + "\n")
    else:
        for _ in range(size):
            print(next(seq_source))

if __name__ == "__main__":
    main()
