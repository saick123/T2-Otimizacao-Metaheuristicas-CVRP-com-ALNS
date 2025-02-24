
import sys
import subprocess
import time as time_bib
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


def search_hyper_paramenters(program_name, save_path = './', programs_number = 1, number_iterations=1, time=300, verbose=0, figure=0):
    
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
            ##########################################################################
            command_args = [
                f'q_max={q_max}',
                f'r={r}',
                f'save_path={save_path}',
                f'number_iterations={number_iterations}',
                f'time={time}',
                f'verbose={verbose}',
                f'figure={figure}',
                f'targets={targets_str}',
            ]
    
            ## q_max, r, save_path, targets, number_iterations
            command_line = ["python3", program_name] + command_args
            
            processo = subprocess.Popen(command_line, stdout=subprocess.PIPE, text=True)
            processes.append(processo)
            #time.sleep(1)
            print(f'processo {len(processes) - 1} criado ...')
    
    
    for i in range(len(processes)):
        processes[i].wait() ## espera todos terminarem
        print(f'processo {i} finalizado ...')
    processes = [] ## reseta lista de processos
                    

def generate_results(program_name, save_path = './', programs_number = 1, number_iterations=1, q_max = 0.15, r=0.3, time=300, verbose=0, figure=0):
    
    total_targets = []  
    for _, paths in files:
        for vrp, _ in paths:
            total_targets.append(vrp)
    
    
    N = len(total_targets) // programs_number
    total_set = set(total_targets)
    j = 0
    targets_for_program = []
    for i in range(1, programs_number):
        targets_for_program.append(total_targets[j: i * N])
        j = i * N
        
    targets_for_program.append(total_targets[j:])
    
    check_set = set()    
    for l in targets_for_program:
        check_set.update(set(l))
    
    if check_set == total_set:
        print('OK')
    else:
        print('WRONG DIVISION')
        raise('dividou as tarefas errado')
    
    command_args = [
        f'q_max={q_max}',
        f'r={r}',
        f'save_path={save_path}',
        f'number_iterations={number_iterations}',
        f'time={time}',
        f'verbose={verbose}',
        f'figure={figure}',
    ]
    
    processes = []
    for targets in targets_for_program:
        
        targets_str = ",".join(targets)
        
        command_args.append(
            f'targets={targets_str}'
        )
        command_line = ["python3", program_name] + command_args
            
        processo = subprocess.Popen(command_line, stdout=subprocess.PIPE, text=True)
        processes.append(processo)

        print(f'processo {len(processes)} criado ...')
    
    time_bib.sleep(5)
    for i in range(len(processes)):
        processes[i].wait() ## espera todos terminarem
        print(f'processo {i+1} finalizado ...')
    
    return 

if __name__ == '__main__':
    
    
    generate_results('app.py', 'resultados_finais_round', 
                     programs_number=4, number_iterations=5,
                     q_max=0.15, r=0.3, time=300,
                     verbose=0, figure=0)
    
    """
    search_hyper_paramenters('app.py', 'resultados_hp_com_round', programs_number=4, 
                             number_iterations=1, time=300, verbose=0, figure=0)
    """