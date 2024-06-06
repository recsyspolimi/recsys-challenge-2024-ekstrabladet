RESIDUAL_DICT = {'residual_0': 0.651223860703235,
                 'residual_1': 0.019967703328025423, 
                 'residual_2': 0.026876955976902865, 
                 'residual_3': 0.20015432836563146, 
                 'residual_4': 0.999285298435996}

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