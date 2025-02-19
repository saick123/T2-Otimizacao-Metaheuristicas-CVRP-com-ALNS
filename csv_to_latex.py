import sys
import csv

class ArgsException(Exception):
    def __init__(self):
        super().__init__()
    def __str__(self) -> str:
        script_name = str(__file__).split('/')[-1]
        return f'Error, missing command line arguments. Try again using this format ==> python3 {script_name} read_path=\'str\' save_path=\'str\''

def csv_to_latex(csv_file, separator = ';'):
    with open(csv_file, 'r') as in_file:

        headers = [x.strip() for x in in_file.readline().split(separator)]
        num_columns = len(headers)
        column_format = ''.join(["c|" for _ in range(num_columns - 1)] + ["c"])
        
        latex_table = "\\begin{tabular}" f"{{{column_format}}}\n" + " & ".join(headers) + " \\\\\n\\hline\n"
        for line in in_file:
            values = [x.strip() for x in line.split(separator)]
            latex_table += " & ".join(values) + " \\\\\n"
        latex_table += "\\end{tabular}"
    return latex_table

def parse_args() -> dict:    
    args = {arg.split("=")[0]: arg.split("=")[1] for arg in sys.argv[1:]}
    return args

def handle_args(args: dict):
    expected_args=['read_path', 'save_path']
    for exp_arg in expected_args:
        if exp_arg not in args:
            raise ArgsException()
    return
    
def main():    
    args = parse_args()
    handle_args(args)
    read_path = args['read_path']
    save_path = args['save_path']
    
    latex_table = csv_to_latex(read_path)
    with open(save_path, 'w') as out_file:
        print(latex_table, file=out_file)
    return 

if __name__ == '__main__':
    main()