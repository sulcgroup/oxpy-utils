from typing import Iterable, Union, Generator

from nupack import *
import numpy as np

from pathlib import Path
import os
import argparse
from Bio.SeqUtils import MeltingTemp as mt

from .seqgen import count_sequences, generate_unique_sequence
from ..structure_editor.dna_structure import rc

def main():
    hlen, gc_count, output_sequences_file, celsius, tolerance, mt_algo, size = ensure_correct_arg_input()

    final_candidate = generate_same_tm_seqs(celsius, gc_count, hlen, mt_algo, size, tolerance)
    with open(output_sequences_file, 'w') as f:
        for key, value in final_candidate.items():
            f.write(f'{key}\n')

    return 0


def generate_same_tm_seqs(celsius: float,
                          gc_count: int,
                          hlen: int,
                          mt_algo: str,
                          size: int,
                          tolerance: float) -> dict[str, float]:
    """
    generates a set of same-size sequences
    """
    generated_sequences = seqgen(hlen, gc_count, size)
    seq_input = list(generated_sequences)
    # create temporary list for sequences input, populate it with reverse-compliments of input sequences
    seqs_input_temp = []  # "temporary" not "temperature"
    for seq in seq_input:
        if seq not in seqs_input_temp:
            seqs_input_temp.append(rc(seq))
    # screen list for forbidden patterns
    # TODO: THIS SHOULD BE APPLIED BEFORE `size` ARGUEMENT
    seqs_input_temp = [seq for seq in seqs_input_temp if seq_check(seq)]
    print(f'{len(seqs_input_temp)} compatible sequences found with handle length of {hlen} and GC count of {gc_count}')
    print(f'Filtering sequences with melting temperature of {celsius}C +- {tolerance}C')
    # screen sequences for correct melting temperature
    final_candidate = melting_temp_screen(seqs_input_temp, celsius, tolerance, mt_algo)
    print(f'{len(final_candidate)} sequences found within tolerance')
    return final_candidate

def generate_same_tm_seqs_to_count(celsius: float,
                                gc_count: int,
                                sequence_length: int,
                                mt_algo: str,
                                num_seqs_to_generate: int,
                                tolerance: float,
                                chunk_size: int=100) -> dict[str, float]:
    """
    should function the same as the above, but will generate sequences until it has found the requested number instead
    of the imprecise "get a bunch and check" approach
    note: may go slightly over the desired count
    """
    # create sequence generator
    good_sequences: dict[str, float] = {}
    while len(good_sequences) < num_seqs_to_generate:
        # grab chunks of seq
        seqs = []
        # construct generator mid loop so we can skip existing sequence reverse compliments
        generator: Generator = generate_unique_sequence(sequence_length, gc_count, set(good_sequences.keys()))
        while len(seqs) < chunk_size:
            try:
                seqs.append(next(generator))
               # allow a chunk that overlaps end of generator
            except StopIteration:
                break
        generator_out_of_sequences = len(seqs) < chunk_size
        # i don't fully understand what we're trying to do here
        seqs_input_temp = []  # "temporary" not "temperature"
        for seq in seqs:
            if seq not in seqs_input_temp:
                seqs_input_temp.append(rc(seq))
        seqs_input_temp = [seq for seq in seqs_input_temp if seq_check(seq)]
        # screen sequences for correct melting temperature
        good_sequences.update(melting_temp_screen(seqs_input_temp, celsius, tolerance, mt_algo))
        if len(good_sequences) < num_seqs_to_generate and generator_out_of_sequences:
            raise Exception(f"Cannot generate {num_seqs_to_generate} "
                            f"sequences of length {sequence_length} with GC count {gc_count} "
                            f"and Tm {celsius}+={tolerance}! Found {len(good_sequences)} sequences.") # todo custom exception
    assert all(rc(seq) not in good_sequences for seq in good_sequences.keys())
    return good_sequences


def parse_arguments() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description="Generate sequences with the same melting temperature, "
                                                 "GC content, and length")
    parser.add_argument('-l',
                        '--hlen',
                        metavar='hlen',
                        type=int,
                        help='Number of nucleotides in the handle EX: 8')
    parser.add_argument('-g',
                        '--gc_count',
                        metavar='gc_count',
                        type=int,
                        help='Number of g or c nucleotides in the handle (Use -1 to allow any gc count) EX: 4')
    parser.add_argument('-o', '--output_sequences_file',
                        metavar='output_sequences_file',
                        type=str,
                        help='Location to save optimum sequences EX: ./example/optimum_sequences.txt')
    parser.add_argument('-c',
                        '--celsius',
                        metavar='celsius',
                        type=float,
                        help='Melting temperature of output sequences at EX: 30')
    parser.add_argument('-t',
                        '--tolerance',
                        metavar='tolerance',
                        type=float,
                        help='Tolerence of the melting temperature window defaults to: +- 0.5C')
    parser.add_argument('-a', '--mt_algo',
                        metavar='mt_algo',
                        type=str,
                        help='By default only returns sequences both Nupack and Biopython agree '
                             'to at 10nM DNA concentration, 0.05M NaCl, and 0.0125M Mg EX: (both | biopy | nupack)')
    parser.add_argument('-s', '--size',
                        metavar='size',
                        type=int,
                        help='Number of randomly generated sequences, by default it generates all possible'
                             ' sequences EX: 50000')
    parser.add_argument('--force_overwrite',
                        action='store_true',
                        help='Force overwrite of and existing output sequence file.')
    args = parser.parse_args()
    return args, parser


def ensure_correct_arg_input() -> tuple[int, int, Path, float, float, str, int]:
    """
    validates that args produced by parse_arguements are correct
    todo: my god make this a dataclass or something
    Returns: a tuple of arguements:
            - desired length of sequence,
            - number of GCs in the seqence
            - file to write sequences
            - melting temperature in degrees C
            - tolerance for deviations in melting tempereature
            - algotithm to use when computing Tm (string identifier)
            - number of sequences to produce
    """
    args, parser = parse_arguments()

    if not args.hlen:
        parser.print_help()
        raise Exception("\nPlease specify the handle length.")
    else:
        hlen = int(args.hlen)

    if not args.gc_count:
        parser.print_help()
        raise Exception("\nPlease specify the number of G or C nucleotides in the handle.")
    else:
        gc_count = int(args.gc_count)
        if gc_count > hlen:
            raise ValueError("GC count cannot be greater than the sequence length")

    if not args.celsius:
        parser.print_help()
        raise Exception("\nPlease specify the temperature to filter the sequences by.")
    else:
        celsius = float(args.celsius)

    # Ensure that the output sequences file is specified and does not already exist
    if not args.output_sequences_file:
        output_sequences_file = Path(f'filtered_{hlen}len_{gc_count}gc_{celsius}c.txt')
        if os.path.exists(output_sequences_file) and not args.force_overwrite:
            raise Exception(
                "An output file already exists:\n{output_sequences_file}\nUse --force_overwrite to overwrite the file")
    else:
        output_sequences_file = Path(args.output_sequences_file)
        if os.path.exists(output_sequences_file) and not args.force_overwrite:
            raise Exception(
                "An output file already exists:\n{output_sequences_file}\nUse --force_overwrite to overwrite the file")
    if output_sequences_file.suffix != '.txt':
        raise Exception("The output file must be a text file")

    if not args.tolerance:
        tolerance = 0.5
    else:
        tolerance = float(args.tolerance)

    if not args.mt_algo:
        mt_algo = 'both'
    else:
        mt_algo = args.mt_algo
        if mt_algo not in ['both', 'biopy', 'nupack']:
            raise Exception("The melting temperature algorithm must be either 'both', 'biopy', or 'nupack")

    if not args.size:
        size = None
    else:
        size = int(args.size)

    return hlen, gc_count, output_sequences_file, celsius, tolerance, mt_algo, size


def seqgen(hlen: int, gc_count: int, size: Union[int, None]) -> set[str]:
    generated_sequences = set()
    print("Sequence length: %s; GC count: %s" % (hlen, gc_count))
    total_sequences = count_sequences(hlen, gc_count)
    print("Total sequence possibilities: %d" % total_sequences)

    if size is None:
        size = total_sequences
    elif size > total_sequences:
        size = total_sequences
    print(f"Generating {size} sequences or {size / total_sequences * 100:.2f}% of all possible sequences")

    seq_generator = generate_unique_sequence(hlen, gc_count)
    return {next(seq_generator) for _ in range(size)}


def seq_check(seq: str) -> bool:
    """
    ???
    I imagine there is a reason this returns int and not bool
    :param seq: a DNA sequence
    """
    # complocal = rc(seq)
    # if complocal == seq: # don't want homopolymers 
    #     return 0

    if 'AAAA' in seq or 'TTTT' in seq or 'GGGG' in seq or 'CCCC' in seq:
        return False
    elif 'AAAAA' in seq or 'TTTTT' in seq or 'GGGGG' in seq or 'CCCCC' in seq:
        return False
    elif 'AAAAAA' in seq or 'TTTTTT' in seq or 'GGGGGG' in seq or 'CCCCCC' in seq:
        return False
    elif 'AAA' in seq or 'TTT' in seq or 'GGG' in seq or 'CCC' in seq:  # newly added
        return False
    # elif 'A' == seq[0] or 'T' == seq[0] or 'A' == seq[-1] or 'T' == seq[-1]: #newly added 
    #     return 0
    # UNDER WHAT CIRCUMSTANCES???
    elif 'A' not in seq or 'T' not in seq or 'G' not in seq or 'C' not in seq:
        return False
        # elif seq[0]=='G' or seq[0]=='C' or seq[-1]=='G' or seq[-1]=='G':
    #     return 1
    else:
        return True

def melting_temp_screen(seqs: list[str],
                        celsius: float,
                        tolerance: float,
                        mt_algo: str) -> dict[str, float]:
    """
    :param seqs: sequences to screen
    :param celsius: desired melting temperature in degrees celsius
    :param tolerance: tolerance for deviations from Tm
    :mt_algo: melting temperature algorithm to use. options: 'both', 'biopy', 'nupack, 'nupack'
    """
    if (mt_algo == 'both') or (mt_algo == 'biopy'):
        biopy_final_candidate = {}
        for seq in seqs:
            bio_temp = mt.Tm_NN(seq, dnac1=10, dnac2=10, Na=50, K=0, Tris=0, Mg=12.5, dNTPs=0, saltcorr=7)
            if (bio_temp > celsius - tolerance) and (bio_temp < celsius + tolerance):
                biopy_final_candidate[seq] = bio_temp

        if mt_algo == 'biopy':
            return biopy_final_candidate

        bioseqs = list(biopy_final_candidate.keys())

        unbound_fraction = get_unbound_fraction(bioseqs, celsius)

        temp_correct_seqs = {}
        for seq, frac in zip(bioseqs, unbound_fraction):
            value = frac
            if (value > 0.45) and (value < 0.55):
                temp_correct_seqs[seq] = value

        temp_window = tolerance * 5
        temp_range = np.linspace(celsius - temp_window, celsius + temp_window, 21)
        all_seqs = list(temp_correct_seqs.keys())
        melting_temperatures, unbound_fractions = compute_melting_temp(all_seqs, temp_range=temp_range)

        final_candidate = {}
        for seq, tp in melting_temperatures.items():
            if (tp > celsius - tolerance) and (tp < celsius + tolerance):
                final_candidate[seq] = tp

        return final_candidate
    # TODO: it looks like the 'nupack' and 'biopy' code is identical here?
    elif mt_algo == 'nupack':
        unbound_fraction = get_unbound_fraction(seqs, celsius)

        temp_correct_seqs = {}
        for seq, frac in zip(seqs, unbound_fraction):
            value = frac
            if (value > 0.45) and (value < 0.55):
                temp_correct_seqs[seq] = value

        temp_window = tolerance * 5
        temp_range = np.linspace(celsius - temp_window, celsius + temp_window, 21)
        all_seqs = list(temp_correct_seqs.keys())
        melting_temperatures, unbound_fractions = compute_melting_temp(all_seqs, temp_range=temp_range)

        final_candidate = {}
        for seq, tp in melting_temperatures.items():
            if (tp > celsius - tolerance) and (tp < celsius + tolerance):
                final_candidate[seq] = tp

    return final_candidate


def create_tube(seq: str, temp, my_model) -> Tube:
    """
    creates a nupack Tube object containing the provided sequence and its reverse compliment
    """
    seqs = [seq, rc(seq)]
    strands = [Strand(seq, name=seq) for seq in seqs]

    c1 = Complex([strands[0]])
    c2 = Complex([strands[1]])
    c3 = Complex([strands[0], strands[1]])

    t1 = Tube(strands={strands[0]: 1e-8, strands[1]: 1e-8}, complexes=SetSpec(include=[c1, c2, c3]), name=f'{seq}')

    return t1


def get_unbound_fraction(all_seqs: Iterable[str], temp: float) -> list[float]:
    my_model = Model(material='dna', celsius=temp, sodium=0.05, magnesium=0.0125)

    all_tubes = []
    for seq in all_seqs:
        all_tubes.append(create_tube(seq, temp, my_model))

    tube_results = tube_analysis(tubes=all_tubes, model=my_model)

    unbound_fraction = []
    for tube in tube_results.fields[0].values():
        concs = tube.complex_concentrations
        names = [c.name for c in concs]
        bound = 0
        unbound = 0
        for conc, name in zip(concs, names):
            name_split = name.split('+')
            if len(name_split) > 1:
                bound += concs[conc]
            else:
                unbound += concs[conc]
        unbound /= 2
        unbound_fraction.append(unbound / (unbound + bound))

    return unbound_fraction


def compute_melting_temp(all_seqs: Iterable[str], temp_range: np.ndarray=np.linspace(1, 100, 1)) -> tuple[dict[str, float],
                                                                                                          dict[str, float]]:
    """
    computes melting temperatures for the provided sequence
    """
    unbound_fractions = {seq: [] for seq in all_seqs}

    for temp in temp_range:
        seq_wise_fractions = get_unbound_fraction(all_seqs, temp)
        for seq, frac in zip(all_seqs, seq_wise_fractions):
            unbound_fractions[seq].append(frac)

    melting_temperatures = {}
    for seq, all_fracs in unbound_fractions.items():
        melt = temp_range[np.abs(np.array(all_fracs) - 0.5).argmin()]
        melting_temperatures[seq] = melt

    return melting_temperatures, unbound_fractions


if __name__ == '__main__':
    main()
