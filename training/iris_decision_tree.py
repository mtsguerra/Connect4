import pandas as pd
from ai_alg.ID3 import ID3Tree

# Discretização pelas frequências, divide igualmente pelo numero de bins
def equal_frequency_discretization(df, columns, bins=3):
    result = df.copy()
    for col in columns:
        result[col] = pd.qcut(result[col], q=bins, labels=['low', 'mid', 'high'])
    return result



if __name__ == "__main__":
    df = pd.read_csv("Connect4-main/training/iris.csv")
    features = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
    target = 'class'

    # Discretização
    df_discretized = equal_frequency_discretization(df, features, bins=3)

    # Divisão em treino e teste
    train = df_discretized.sample(frac=0.8, random_state=1)
    test = df_discretized.drop(train.index)

    # Construir árvore
    tree = ID3Tree(max_depth=None)
    tree.fit(train[features], train[target])
    tree.predict(test[features])

    # Medir acurácia
    preds = tree.predict(test[features])

    # Compara direto com a coluna alvo
    accuracy = (preds == test[target]).mean()
    print(f"Acurácia: {accuracy:.2%}")