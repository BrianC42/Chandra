'''
Created on Sep 11, 2020

@author: Brian

command line
pathin: path to input files
files: list of files to use as input
fields: list of data fields to include in the output
pathout: path to output file
output: output file name
'''
import argparse

if __name__ == '__main__':
    print("Hello World ...\n")
    
    parser = argparse.ArgumentParser(description='Prepare market data for training.')
    parser = argparse.ArgumentParser(epilog='Intended to be called as part of the learning network training.')
    '''
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max,
                        help='sum the integers (default: find the max)')
    args = parser.parse_args()
    print(args.accumulate(args.integers))
    '''
    parser.add_argument('--pathin', help='path to input files')
    parser.add_argument('--file', action='append', help='file containing enhanced market data')
    parser.add_argument('--field', action='append', help='enhanced market data field')
    parser.add_argument('--pathout', help='path to output file')
    parser.add_argument('--output', help='output file name')

    args = parser.parse_args()
    
    print("parameters:\n %s" % args)
    print("pathin %s" % args.pathin)
    print("files %s" % args.file)
    print("fields %s, %s, %s" % (args.field, args.field[0], args.field[1]))
    print("pathout %s" % args.pathout)
    print("output %s" % args.output)
    
    print("\nGoodbye cruel world")
