# T2-Otimizacao-Metaheuristicas-CVRP-com-ALNS
Esse trabalho é referente a disciplica de Tópicos em Otimização: Meta-heurísticas. O objetivo é estudar a aplicação da meta-heurística ALNS (Adaptive Large Neighborhood Search) no problema CVRP (Capacitated Vehicle Routing Problem).

# Instruções de Execução
- Instale as dependências necessárias com o comando `pip install -r requirements.txt`
- Para rodar o programa, execute o arquivo `app.py` da pasta `src` com a seguinte formatação de linha de comando: 

- `python3 app.py q_max=<float_value> r=<float_value> save_path=<path> targets=<path> number_iterations=<int_value> time=<seconds> verbose=<int_value> figure=<int_value>`
- Os valores ``q_max``=0.15 e ``r``=0.3 foram utilizados nesse trabalho.
- Para as flags ``verbose`` e ``figure``, 1 significa True e 0 False.
- Na flag ``targets``, deve ser passado uma `string` com o `path` de todos os arquivos ``.vrp`` alvo separados por ``,`` (vírgula).
- Por fim, um exemplo de linha de comando para o programa pode ser encontrado no arquivo ``command_line_example.txt``
- O artigo do trabalho para mais detalhes pode ser encontrado também no repositório na pasta ``pdf`` com o nome de ``article.pdf``
