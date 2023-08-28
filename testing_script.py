import numpy as np
import pandas as pd
import parser
import model_test

data, indexes = parser.build_data_df('/Users/stanislavananyev/PycharmProjects/GPN/Wells_data.xlsx',
                            '/Users/stanislavananyev/PycharmProjects/GPN/NLM', drop_zero_g=True) ##Drop zero g ставить на True, если без интерполяции делать
# data_after_inter = parser.interpolate_data(data) ## Интерполяция недостающих значений, сначала лучше с ней посчитать, потом можно еще раз без нее

bad_models = [] ## список с индексами моделей не прошедщих тест
for i in range(0, len(data)):## в data передаешь либо сразу список data (без интеполяции), либо тот список, которому приравнял интеполированные данные
    good_model = 1
    counter = 0
    index = indexes[i]
    while good_model == 1:
        score, flag = model_test.train_and_check(data[i], '/Users/stanislavananyev/PycharmProjects/GPN/NLM/NLM_models/nlm', index, 0.1, counter) ## ареса укажи сам какие хочешь, в данном случае указывается папка с моделями, нде модели будт именоваться nlm0, nlm1, nlm2...
        print("Iteration # ", counter)
        print(score)
        print(flag, '\n')
        good_model = flag
        counter += 1
        if good_model == 0:
            print('Good model: ', index, ", score= ", score)
            with open('/Users/stanislavananyev/PycharmProjects/GPN/NLM/NLM_scores.txt', 'a') as sc_f:
                sc_f.write('Model: {}, score = {}\n'.format(index, score))## схранение файла со скором и индексом моделей прошедших тест
        if counter > 50:
            bad_models.append(index)
            with open('/Users/stanislavananyev/PycharmProjects/GPN/NLM/NLM_bad_models.txt', 'a') as bm:
                for item in bad_models:
                    bm.write("%d\n" % item)
            break
#

# ##


