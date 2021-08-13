#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import utils_NNs as utils




if __name__ == "__main__":
    
    data = read_csv_file(path)
    best_score, best_params= find_best_param_in_grid(model,learning_rates,batch_sizes, X_train,y_train)
    batch_size = best_params.get('batch_size')
    learning_rate = best_params.get('learning_rate')
    train_model = train_model_n_fold_cv(mlp_elmo_model(learning_rate), batch_size, data, n_split=5, test_size=0.2)
    model, acc, report = evaluate_model(trained_model, test_set)

    print(acc, report)

