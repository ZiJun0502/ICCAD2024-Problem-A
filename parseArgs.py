import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process optimization parameters.')

    # Adding arguments as per the command example
    parser.add_argument('-cost_function', type=str, required=True, help='Specify the cost function to be used')
    parser.add_argument('-library', type=str, required=True, help='Specify the library file to be used')
    parser.add_argument('-netlist', type=str, required=True, help='Specify the netlist file to be used')
    parser.add_argument('-output', type=str, required=True, help='Specify the output file name')

    # Parsing the arguments
    args = parser.parse_args()

    # Printing the arguments (or process them as needed)
    print(f"Cost Function: {args.cost_function}")
    print(f"Library: {args.library}")
    print(f"Netlist: {args.netlist}")
    print(f"Output: {args.output}")

    # Return the parsed arguments for further processing if needed
    return args