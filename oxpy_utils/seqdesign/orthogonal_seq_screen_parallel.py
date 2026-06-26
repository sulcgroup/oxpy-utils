# Import NUPACK Python module
from typing import Collection, Union

from nupack import Model
import argparse
from pathlib import Path
import os

from ..structure_editor.dna_structure import rc
from .sequence_optimization_functions import load_ensemble, fill_dGs, ddG_optimization_algorithm

def main():
    args, parser = parse_arguments()

    # todo: this should be a dataclass ngl
    candidate_sequences_file, num_seq_output, \
    optimization_steps, dG_file, output_sequences_file, spacer, \
    score_no_opt, celsius, filter_struct, sequence_constraint_file = ensure_correct_arg_input(args, parser)
    
    print('\nOutput strands in 5-3 direction, and assume input is the same')

    # Define physical model for nupack 
    my_model = Model(material='dna', celsius=celsius)

    #load seqs and add its RC
    seqs_input = load_ensemble(candidate_sequences_file)
    
    # Load constraint sequences if provided
    if sequence_constraint_file is not None:
        constraint_seqs = load_ensemble(sequence_constraint_file)
        # Ensure constraints are in the candidate sequences
        missing_seqs = set(constraint_seqs) - set(seqs_input)
        if missing_seqs:
            raise Exception(f"The following constraint sequences are not in the candidate sequences: {missing_seqs}")
    else:
        constraint_seqs = []

    best_distance, optimal_sequence_set = compute_orthogonal_seqs(constraint_seqs, dG_file, filter_struct, my_model,
                                                                  num_seq_output, optimization_steps, score_no_opt,
                                                                  seqs_input, spacer)

    # threshold = evaluate_threshold(optimal_sequence_set, my_model, num_seq_output, dGs_filled)
    # print(threshold)
    
    with open(output_sequences_file, 'w') as f:
        for seq in optimal_sequence_set:
            f.write(f'{seq}\n')
            
    print(f"\nBest subset of {num_seq_output} sequences including complements: {optimal_sequence_set}\nThreshold: {best_distance}\nOutput saved to {output_sequences_file}")
    
    return 0


def compute_orthogonal_seqs(constraint_seqs: list[str],
                            dG_file: Path,
                            filter_struct: int,
                            my_model: Model,
                            num_seq_output: int,
                            optimization_steps: int,
                            score_no_opt: bool,
                            seqs_input: list[str],
                            spacer: Union[str, None] = None):
    """
    Compute orthogonal sequences
    Args:
        constraint_seqs: a list of sequence which *must* be part of the solution
        dG_file: path to a file in which to store the dGs
        filter_struct: the maximum number of allowed secondary structures (really? 3?)
        my_model: a nupack mode object
        num_seq_output: requested number of sequences
    """
    assert all(rc(seq) in seqs_input for seq in seqs_input), "set of "
    # Find the orthogonal seqs
    dGs_filled = fill_dGs(seqs_input,
                          my_model,
                          dG_file,
                          spacer,
                          filter_struct,
                          constraint_seqs)
    # run optimization algorithm
    optimal_sequence_set, best_distance = ddG_optimization_algorithm(dGs_filled,
                                                                     num_seq_output,
                                                                     optimization_steps,
                                                                     score_no_opt,
                                                                     constraint_seqs)
    if len(constraint_seqs) > 0:
        assert len(optimal_sequence_set) == num_seq_output * 2 + len(constraint_seqs), \
            (f"Optimal sequence output count {len(optimal_sequence_set)} does not match "
             f"expected {num_seq_output * 2 + len(constraint_seqs)}")
    else:
        assert len(optimal_sequence_set) == num_seq_output * 2, \
            (f"Optimal sequence output count {len(optimal_sequence_set)} does not match "
             f"expected {num_seq_output * 2}")
    if spacer is not None:
        optimal_sequence_set = [spacer + seq for seq in optimal_sequence_set]
    return best_distance, optimal_sequence_set


#Enable command line arguments
def parse_arguments() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description="Finds the list of n sequences that have as high (=unfavorable) dG"
                                                 " with all other seqs except its own complement. Assumes the "
                                                 "sequences in the file are in the 5' to 3' order")
    parser.add_argument('-f',
                        '--candidate_sequences_file',
                        metavar='candidate_sequences_file',
                        type=str,
                        help='File containing the candiate sequeunces EX: ./example/seq1K.txt')
    parser.add_argument('-n',
                        '--num_seq_output',
                        metavar='num_seq_output',
                        type=int,
                        help='The number of sequence pairs (seq & complement) to be output EX: 6')
    parser.add_argument('-s',
                        '--optimization_steps',
                        metavar='optimization_steps',
                        type=int,
                        help='The number of optimization steps to be performed EX: 100')
    parser.add_argument('-g',
                        '--dG_file',
                        metavar='dG_file',
                        type=str,
                        help='Save dG_file if one does not exist, otherwise load file EX: ./example/dGs_filled.pkl')
    parser.add_argument('-o',
                        '--output_sequences_file',
                        metavar='output_sequences_file',
                        type=str,
                        help='Location to save optimum sequences EX: ./example/optimum_sequences.txt')
    parser.add_argument('-t',
                        '--spacer',
                        metavar='spacer',
                        type=str,
                        help='Nuleotides to add 5` end of all sequences EX: TTTTTTTTTTTT')
    parser.add_argument('-c',
                        '--celsius',
                        metavar='celsius',
                        type=float,
                        help='Temperature to test the sequences at EX: 20')
    parser.add_argument('--sequence_constraint_file',
                        metavar='sequence_constraint_file',
                        type=str,
                        help='File containing sequences that must be a part of the solution. If the file does not contain a sequences complment, it will still be added to the solution by default \
                              To us this option it is important to know:\n \
                              1. The sequences need to also be in the candidate_sequences_file\n \
                              2. If you use the spacer option, the spacer will be added to the constraint sequences\n \
                              3. The constraint sequences do not count towards the num_seq_output pairs (i.e if you have 10 constrained seqs and -n 10 it will output 20 pairs)\n \
                              EX: ./example/must_have_overhangs.txt')
    parser.add_argument('--filter_struct',
                        metavar='filter_struct',
                        type=float,
                        help='The number of allowed secondary strucutre base pairs per sequence EX: 0')
    parser.add_argument('--score_no_opt',
                        action='store_true',
                        help='Score the sequences without optimization')
    parser.add_argument('--force_overwrite',
                        action='store_true',
                        help='Force overwrite of and existing output sequence file.')
    args = parser.parse_args()
    return args, parser


def ensure_correct_arg_input(args: argparse.Namespace, parser: argparse.ArgumentParser):
    # Ensure that the candidate sequences file is specified and exists
    if not args.candidate_sequences_file:
        parser.print_help()
        raise Exception("\nPlease specify the file containing the candidate sequences using -f or --candidate_sequences_file.")
    else:
        if not os.path.exists(args.candidate_sequences_file):
            parser.print_help()
            raise Exception("\nThe specified file containing the candidate sequences does not exist")
        else:
            with open(args.candidate_sequences_file, 'r') as f:
                for line in f:
                    # If a line contains a sequence that is not a valid DNA or RNA sequence, print an error message and exit
                    if not set(line.strip().upper()).issubset({'A', 'C', 'G', 'T', 'U'}):
                        parser.print_help()
                        raise Exception("\nPlease ensure that the file contains only valid DNA or RNA sequences")
    candidate_sequences_file = Path(args.candidate_sequences_file)
    
    # Ensure that the number of sequences to be output is specified and is an integer
    if not args.num_seq_output:
        raise Exception("Please specify the number of sequences to be output using -n or --num_seq_output")
    assert type(args.num_seq_output) == int, "The number of sequences to be output must be an integer"
    num_seq_output = args.num_seq_output

    # Ensure that the number of optimization steps is specified and is an integer
    if not args.optimization_steps:
        optimization_steps = 1000
    else:
        assert type(args.optimization_steps) == int, "The number of optimization steps must be an integer"
        optimization_steps = args.optimization_steps
    
    # Ensure that the dG file is specified and exists
    if not args.dG_file:
        dG_file = Path(f'dGs_{candidate_sequences_file.stem}_{optimization_steps}_steps.pkl')
    else:
        if not args.dG_file.endswith('.pkl'):
            parser.print_help()
            raise Exception("The specified file containing the dG values must be a pickle file")
        dG_file = Path(args.dG_file)
        
    # Ensure that the output sequences file is specified and does not already exist
    if not args.output_sequences_file:
        output_sequences_file = Path(f'optimum_sequences_{candidate_sequences_file.stem}_{optimization_steps}_steps.txt')
        if os.path.exists(output_sequences_file) and not args.force_overwrite:
            raise Exception("An output file already exists:\n{output_sequences_file}\nUse --force_overwrite to overwrite the file")
    else:
        output_sequences_file = Path(args.output_sequences_file)
        if os.path.exists(output_sequences_file) and not args.force_overwrite:
            raise Exception("An output file already exists:\n{output_sequences_file}\nUse --force_overwrite to overwrite the file")
    if output_sequences_file.suffix != '.txt':
        raise Exception("The output file must be a text file")
    
    
    if not args.score_no_opt:
        score_no_opt = False
    else:
        score_no_opt = True
    
    
    if not args.spacer:
        spacer = None
    else:
        spacer = args.spacer
        assert set(spacer.upper()).issubset({'A', 'C', 'G', 'T', 'U'}), "The spacer must be a valid DNA or RNA sequence"
    
    if not args.celsius :
        celsius  = 20
    else:
        celsius = float(args.celsius)
        
    if not args.filter_struct:
        filter_struct = 0
    else:
        filter_struct = args.filter_struct
        
    if not args.sequence_constraint_file:
        sequence_constraint_file = None
    else:
        sequence_constraint_file = Path(args.sequence_constraint_file)
        if not os.path.exists(sequence_constraint_file):
            parser.print_help()
            raise Exception("\nThe specified file containing the sequence constraints does not exist")
        else:
            with open(sequence_constraint_file, 'r') as f:
                for line in f:
                    # If a line contains a sequence that is not a valid DNA or RNA sequence, print an error message and exit
                    if not set(line.strip().upper()).issubset({'A', 'C', 'G', 'T', 'U'}):
                        parser.print_help()
                        raise Exception("\nPlease ensure that the file contains only valid DNA or RNA sequences")

    return candidate_sequences_file, num_seq_output, optimization_steps, dG_file, output_sequences_file, spacer, score_no_opt, celsius, filter_struct, sequence_constraint_file

    
if __name__ == '__main__':
    main()