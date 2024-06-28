RESIDUAL_DICT = {'residual_0': 0.5919690814492665, 
                 'residual_1': 0.0012995886278801857, 
                 'residual_2': 0.08266743331619494, 
                 'residual_3': 0.5029040319637748, 
                 'residual_4': 0.7179375798846832}
# weigths with six models tuned on large dataset
if __name__ == '__main__':
    residual_list = [res for res in RESIDUAL_DICT.values()]
    n_weights = len(residual_list)
    
    residual = 1.0
    
    weights = []
    for i in range(n_weights):
        assert residual_list[i] >= 0 and residual_list[i] <= 1
        weights.append(residual * residual_list[i])
        residual = residual - residual*residual_list[i]
    weights.append(residual)
    
    print(weights)