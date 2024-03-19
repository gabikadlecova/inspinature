import os
from sklearn.model_selection import ParameterGrid


# ideally load from some json
grid_data = {
   'cx_pb': [0.1, 0.25, 0.3],
   'mut_pb': [0.2, 0.3],
   'pop_size': [100, 200]
}


if __name__ == "__main__":
    for params in ParameterGrid(grid_data):
        # OR save it to a dataframe, then load ith row in the program
        out_path = os.path.join('test', '_'.join(f'{k}-{v}' for k, v in params.items()), '.json')
        params['out_path'] = out_path
        
        pop_size = params.pop('pop_size')
        
        print(pop_size, end=' ')
        print(" ".join([f"--{k} {v}" for k, v in params.items()]))
