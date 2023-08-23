import parser
import model_test
from tqdm import tqdm

with open("/Users/stanislavananyev/PycharmProjects/GPN/LugBadModels.txt") as f:
    bad_models = [int(x) for x in f.read().split()]

grid = model_test.generate_grid()
img_path = '/Users/stanislavananyev/PycharmProjects/GPN/LugFigures'
dictionary_path = '/Users/stanislavananyev/PycharmProjects/GPN/well_dict.pickle'
for i in tqdm(bad_models):
    df = model_test.check_the_model(i, grid)
    model_test.plot_results(df, i, dictionary_path, img_path)