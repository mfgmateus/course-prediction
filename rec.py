import sys

import pandas as pd
import turicreate as tc
from sklearn.model_selection import train_test_split

sys.path.append("..")

initial_data = pd.read_csv('dados_aluno_disciplina.csv')
disciplinas = pd.read_csv('disciplinas.csv')
data = pd.read_csv('dados_aluno_disciplina.csv')

data = data.groupby(['aluno', 'disciplina']).agg({'count': 'count'}).reset_index()


def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['count_dummy'] = 1
    return data_dummy


data_dummy = create_data_dummy(data)


def normalize_data(data):
    df_matrix = pd.pivot_table(data, values='count', index='aluno', columns='disciplina')
    df_matrix_norm = (df_matrix - df_matrix.min()) / (df_matrix.max() - df_matrix.min())
    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_count']
    return pd.melt(d, id_vars=['aluno'], value_name='scaled_count').dropna()


def split_data(data):
    """
    Splits dataset into training and test set.

    Args:
        data (pandas.DataFrame)

    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    """
    train, test = train_test_split(data, test_size=.2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data


data_norm = normalize_data(data)

train_data, test_data = split_data(data)
train_data_dummy, test_data_dummy = split_data(data_dummy)
train_data_norm, test_data_norm = split_data(data_norm)

# constant variables to define field names include:
user_id = 'aluno'
item_id = 'disciplina'
alunos_to_recomend = initial_data.aluno.unique().tolist()
n_rec = 2  # number of items to recommend
n_display = 10  # to display the first few rows in an output dataset
target = 'count'


def model(train_data, name, aluno, disciplina, target, alunos_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data,
                                                 user_id=aluno,
                                                 item_id=disciplina,
                                                 target=target)
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data,
                                                      user_id=aluno,
                                                      item_id=disciplina,
                                                      target=target,
                                                      similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data,
                                                      user_id=aluno,
                                                      item_id=disciplina,
                                                      target=target,
                                                      similarity_type='pearson')
    recom = model.recommend(users=alunos_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model


popularity = model(train_data, 'popularity', user_id, item_id, target, alunos_to_recomend, n_rec, n_display)

cos = model(train_data, 'cosine', user_id, item_id, target, alunos_to_recomend, n_rec, n_display)

pear = model(train_data, 'pearson', user_id, item_id, target, alunos_to_recomend, n_rec, n_display)

models_w_counts = [popularity, cos, pear]
names_w_counts = ['Popularity Model', 'Cosine Similarity', 'Pearson Similarity']

eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)

data_dummy = create_data_dummy(data)
final_model = tc.item_similarity_recommender.create(tc.SFrame(data),
                                                    user_id=user_id,
                                                    item_id=item_id,
                                                    target=target,
                                                    similarity_type='cosine')

recom = final_model.recommend(users=alunos_to_recomend, k=n_rec)
recom.print_rows(n_display)

df_rec = recom.to_dataframe()
# print(df_rec.shape)
df_rec.head()


def create_output(model, to_recommend, n_rec, print_csv=True):
    recomendation = model.recommend(users=to_recommend, k=n_rec)
    df_rec = recomendation.to_dataframe()
    df_rec['recommendedCourses'] = df_rec.groupby([user_id])[item_id] \
        .transform(lambda x: '|'.join(x.astype(str)))
    df_output = df_rec[['aluno', 'recommendedCourses']].drop_duplicates() \
        .sort_values('aluno').set_index('aluno')
    if print_csv:
        df_output.to_csv('cosine_recommendation.csv')
    return df_output


df_output = create_output(cos, alunos_to_recomend, n_rec, print_csv=True)

# print(df_output.shape)
df_output.head()


def student_recomendation(student_id):
    if student_id not in df_output.index:
        print('student not found.')
        return student_id
    return df_output.loc[student_id]


print(student_recomendation(2))
print(student_recomendation(23))