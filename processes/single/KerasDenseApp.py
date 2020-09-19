'''
Created on Jan 31, 2018

@author: Brian
'''
import argparse

if __name__ == '__main__':
    print("... KerasDenseApp ...")
    parser = argparse.ArgumentParser(description='Prepare market data for training.')
    parser = argparse.ArgumentParser(epilog='Intended to be called as part of the learning network training.')
    parser.add_argument('--pathin', help='path to input files')
    parser.add_argument('--file', action='append', help='file containing enhanced market data')
    parser.add_argument('--field', action='append', help='enhanced market data field')
    parser.add_argument('--pathout', help='path to output file')
    parser.add_argument('--output', help='output file name')

    args = parser.parse_args()
    
    print("parameters:\n %s" % args)
    print("pathin %s" % args.pathin)
    print("files %s" % args.file)
    #print("fields %s, %s, %s" % (args.field, args.field[0], args.field[1]))
    print("pathout %s" % args.pathout)
    print("output %s" % args.output)
    
    print("\nDense model trained")
