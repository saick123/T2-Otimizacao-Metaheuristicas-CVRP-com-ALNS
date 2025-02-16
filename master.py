
import sys
import subprocess
import time
import os

def parse_args() -> dict:    
    args = {arg.split("=")[0]: arg.split("=")[1] for arg in sys.argv[1:]}
    return args

folders = ['Vrp-Set-' + c for c in ['A', 'B', 'F']]
files = []
instance = None

for folder in folders:
    innter_folder_path = os.path.join(folder, folder[-1])
    files_path = sorted(list({os.path.join(innter_folder_path, s[:-4]) for s in os.listdir(innter_folder_path)}))
    files_ = ( folder, [(fp + '.vrp', fp + '.sol') for fp in files_path])
    files.append(files_)


def search_hyper_paramenters(programs_number = 1, number_iterations=1):
    
    q_maxes = [0.15, 0.20, 0.30]
    r_values = [0.1, 0.3, 0.5]
    #init_temps = [100, 1000, 10000]

    T = ['A-n32-k5', 'A-n46-k7', 'A-n80-k10', 'B-n50-k8', 'B-n78-k10']
    targets = []
    for _, paths in files:
        for vrp, _ in paths:
            for t in T:
                if t in vrp:
                    targets.append(vrp)
                
    targets_str = ",".join(targets)
    
    ## passar so caminhos, os valores dos hiperparametros, o caminho para salvar os resultados
    processes = []
    for q_max in q_maxes:
        for r in r_values:
            
            if len(processes) == programs_number: ## se já tiver executando o máximo
                for i in range(len(processes)):
                    processes[i].wait() ## espera todos terminarem
                    print(f'processo {i} finalizado ...')
                processes = [] ## reseta lista de processos
            
            file_name = f'q_max-r-{q_max}-{r}.pkl'
            save_path = 'args_results'
            
            ## q_max, r, save_path, targets, number_iterations
            command_line = ["python3", "test_command.py", f"q_max={q_max}", f'r={r}'
                            , f'save_path={save_path}', f'targets={targets_str}', f'number_iterations={number_iterations}']
            
            processo = subprocess.Popen(command_line, stdout=subprocess.PIPE, text=True)
            processes.append(processo)
            #time.sleep(1)
            print(f'processo {len(processes) - 1} criado ...')
    
    
    for i in range(len(processes)):
        processes[i].wait() ## espera todos terminarem
        print(f'processo {i} finalizado ...')
    processes = [] ## reseta lista de processos
                    


if __name__ == '__main__':
    
    
    search_hyper_paramenters(programs_number=5, number_iterations=1)