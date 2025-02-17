import os

## instance, param1(q_max), param2(r), f_sol_optimal, f_mh_min, f_mh_mean, temp_min, temp_mean, gap_min, gap_mean
#"A-n32-k5, 0.2, 0.1, 784, 788, 788.0, 4.190832853317261, 4.190832853317261, 0.00510204081632653, 0.00510204081632653"

if __name__ == "__main__":
    
    folder_path = 'final_results'
    save_name = 'final_results.csv'
    header = 'instance; param1; param2; f_sol_optimal; f_mh_min; f_mh_mean; temp_min; temp_mean; gap_min; gap_mean'
    
    files = os.listdir(folder_path)
    
    lines = []
    files.sort()
    for file in files:
        with open(os.path.join(folder_path, file), 'r') as in_file:
            line = in_file.readline()
            line = line.split(',')
            line = [x.strip() for x in line]
            lines.append(line)
            
    for line in lines:
        for i in range(1, len(line)):
            line[i] = float(line[i])
        for i in range(len(line)):
            if i >=6:
                #line[i] = f'{line[i]:.7}'
                
                line[i] = f'{line[i]:.7}'.replace('.', ',')
            else:
                #line[i] = str(line[i])
                
                line[i] = str(line[i]).replace('.', ',') 
    
    with open(save_name, 'w') as out_file:
        print(header, file=out_file)
        for line in lines:
            str_ = ';'.join(line)
            print(str_, file=out_file)
    
