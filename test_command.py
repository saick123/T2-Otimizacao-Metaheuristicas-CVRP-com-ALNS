

import sys
import os

def parse_args() -> dict:    
    args = {arg.split("=")[0]: arg.split("=")[1] for arg in sys.argv[1:]}
    return args



if __name__ == '__main__':
    
    args = parse_args()
    save_path = args['save_path']
    
    q_max = args['q_max']
    r = args['r']
    with open(os.path.join(save_path, f'qmax-r-{q_max}-{r}.txt'), 'w') as out_file:
        print(args, file=out_file)
