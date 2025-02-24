import os

## instance, param1(q_max), param2(r), f_sol_optimal, f_mh_min, f_mh_mean, temp_min, temp_mean, gap_min, gap_mean
#"A-n32-k5, 0.2, 0.1, 784, 788, 788.0, 4.190832853317261, 4.190832853317261, 0.00510204081632653, 0.00510204081632653"

colunas_args_results = {0, 1, 2, 4}
colunas_final_results = {0, 3, 4, 5, 6, 7, 8, 9}


if __name__ == "__main__":
    
    folder_path = 'resultados_finais_round'
    save_name = 'resultados_finais_round.csv'
    header = 'instance; param1; param2; f_sol_optimal; f_mh_min; f_mh_mean; temp_min; temp_mean; gap_min; gap_mean'
    colunas_filtro = colunas_final_results
    header = header.split(';')
    
    files = os.listdir(folder_path)
    files.sort()
    
    lines = []
    
    for file in files:
        with open(os.path.join(folder_path, file), 'r') as in_file:
            line = in_file.readline()
            line = in_file.readline()
            line = [x.strip() for x in line.split(';')]
            lines.append(line)
           
    lines.insert(0, header) ## colocando header no começo    
    for i, line in enumerate(lines):
        filtered_line = [x for idx, x in enumerate(line) if idx in colunas_filtro]
        lines[i] = filtered_line
    
    for idx, line in enumerate(lines):
        if idx == 0: continue
        for i in range(len(line)):
            if i<=7 and i >=6:
                line[i] = f'{float(line[i]):.7f}'
            elif i == 4 or i == 5:
                line[i] = f'{float(line[i]):.3f}'
            elif i == 3:
                line[i] = f'{float(line[i]):.2f}'
    
    
    for line in lines:
       print(line)
    
    with open(save_name, 'w') as out_file:
        for line in lines:
            str_ = ';'.join(line)
            print(str_, file=out_file)
    
