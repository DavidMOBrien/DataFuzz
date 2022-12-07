import os
import sys
import json
import argparse

def run(runfile):
  with open(runfile,"r") as rnf:
    exec(rnf.read())

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--program",
        help="Run a .py preprocessing program.",
        default=None
    )

    parser.add_argument(
        "--iter",
        type=int,
        help="Maximum number of iteration to run the program.",
        default=1
    )

    parser.add_argument(
        "--runAll",
        help="Run all the available programs (pass 'pre' for this project).",
    )

    args = parser.parse_args()

    if args.program:
        if args.iter > 1:
            for max_iter in range(args.iter):
                run(args.program)
    
    if args.runAll:
        for n in range(1, 8):
            try:
                file_name = "TT"+str(n)+'.py'
                print("----- Running TT%s.py -------- " % str(n))
                run(file_name)
            except (FileNotFoundError):
                print("Ops, couldn't find the program TT %s .py" % str(n))
            except (AttributeError, KeyError, ValueError, NameError) as error:
                print("This program fails with the message: %s" % error)
            else:
                print("Runs properly")


if __name__ == "__main__":
    main()