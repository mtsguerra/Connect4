import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_ai_results(csv_path, output_dir="AIs_statistics"):
    # Cria diretório para salvar os gráficos
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    df['par'] = df['player1_ai'] + " vs " + df['player2_ai']

    # 1. Número médio de jogadas por par de IA
    avg_moves = df.groupby('par')['total_moves'].mean().reset_index()
    plt.figure(figsize=(10,5))
    sns.barplot(x='par', y='total_moves', data=avg_moves)
    plt.xticks(rotation=45)
    plt.ylabel("Média de Jogadas na Partida")
    plt.title("Média de Jogadas para Cada Par de IA")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "media_jogadas_por_par.png"))
    plt.close()
    
    # 2. Distribuição de vitórias/empates por par
    par_resultados = df.groupby(['par', 'winner']).size().unstack(fill_value=0)
    par_resultados_norm = par_resultados.div(par_resultados.sum(axis=1), axis=0)
    par_resultados_norm.plot(kind='bar', stacked=True, figsize=(12,6), colormap='tab20')
    plt.ylabel('Proporção')
    plt.title('Distribuição dos Resultados por Par de IA')
    plt.legend(title='Resultado')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribuicao_resultados_por_par.png"))
    plt.close()
    
    print(f"Gráficos salvos no diretório: {output_dir}")

if __name__ == "__main__":
    plot_ai_results("Connect4-main/training/data/ai_vs_ai_results.csv")  # Altere o nome do arquivo se necessário
