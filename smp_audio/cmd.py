import sys
import argparse

# class smp_audioArgumentParser(argparse.ArgumentParser):

def smp_audioArgumentParser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help='auto command help', dest='mode')

    # autoedit 
    subparser_autoedit = subparsers.add_parser('autoedit', help='autoedit help')
    subparser_autoedit.add_argument("-f", "--filenames", dest='filenames', help="Input file(s) []", nargs = '+', default = [], required=True)
    subparser_autoedit.add_argument("-o", "--outputs", dest='outputs', help="Output types / file types [wav] (wav, pkl, txt)", nargs = '*', default = ['wav'])
    subparser_autoedit.add_argument("-a", "--assemble-mode", dest='assemble_mode',
                        help="Assemble mode [random] (random, sequential)",
                        default='random')
    subparser_autoedit.add_argument("-ax", "--assemble-crossfade", dest='assemble_crossfade', type=int,
                        help="Crossfade duration in assemble [10]",
                        default=10)
    subparser_autoedit.add_argument("-d", "--duration", dest='duration', default=180, type=float, help="Desired duration in seconds [180]")
    subparser_autoedit.add_argument("-ns", "--numsegs", dest='numsegs', default=10, type=int, help="Number of segments for segmentation")
    subparser_autoedit.add_argument("-src", "--sr-comp", dest='sr_comp', default=22050, help="Sample rate for computations [22050]")
    subparser_autoedit.add_argument("-smin", "--seglen-min", dest='seglen_min', default=2, help="Segment length minimum in seconds [2]")
    subparser_autoedit.add_argument("-smax", "--seglen-max", dest='seglen_max', default=60, help="Segment length maximum in seconds [60]")
    subparser_autoedit.add_argument("-w", "--write", dest='write', action='store_true', default=False, help="Write output [False]")

    # autocover
    subparser_autocover = subparsers.add_parser('autocover', help='autocover help')
    subparser_autocover.add_argument("-f", "--filenames", dest='filenames', help="Input file(s) []", nargs = '+', default = [], required=True)
    subparser_autocover.add_argument("-o", "--outputs", dest='outputs', help="Output types / file types [wav] (json, pdf, jpg)", nargs = '*', default = ['json'])
    subparser_autocover.add_argument(
        "-acm", "--autocover-mode", dest='autocover_mode',
        help="autocover mode [feature_matrix] (feature_matrix, recurrence_matrix)",
        default='feature_matrix')

    # automaster
    subparser_automaster = subparsers.add_parser('automaster', help='automaster help')
    subparser_automaster.add_argument("-b", "--bitdepth", dest='bitdepth', default=24, help="Bitdepth for computations [24] (16|24)")
    subparser_automaster.add_argument(
        "-f", "--filenames",
        dest='filenames', help="Input file(s) []",
        nargs = '+', default = [], required=True)
    subparser_automaster.add_argument(
        "-r", "--references",
        dest='references', help="reference file(s) []",
        nargs = '+', default=[], required=True)
    subparser_automaster.add_argument("-o", "--outputs", dest='outputs', help="Output types / file types [wav] (wav)", nargs = '*', default = ['wav'])

    parser.add_argument("-s", "--sorter", dest='sorter', default='features_mt_spectral_spread_mean', help="Sorting feature [features_mt_spectral_spread_mean]")
    parser.add_argument("-r", "--rootdir", type=str, default='./data', help="Root directory to prepend to all working directories [./data]")
    parser.add_argument("--seed", dest='seed', type=int, default=123, help="Random seed [123]")
    parser.add_argument("-v", "--verbose", dest='verbose', action='store_true', default=False, help="Be verbose [False]")
    
    return parser
