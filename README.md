
# 🧾 Modelos Baseados em Árvores (Classificação e Regressão)

## Sobre o Projeto

Este notebook demonstra o uso de **Modelos de Aprendizado de Máquina baseados em Árvores de Decisão**, aplicados a dois problemas distintos:

1. **Classificação** – Diagnóstico de câncer de mama (dataset *Breast Cancer Wisconsin*).  
2. **Regressão** – Predição do consumo de combustível (*milhas por galão*) com base em características de automóveis (*dataset Auto MPG*).

O objetivo é compreender:
- Como funcionam as **árvores de decisão** para **classificação** e **regressão**.  
- Como ajustar **hiperparâmetros** como profundidade e critério de divisão.  
- Como avaliar **acurácia** e **erro médio quadrático (RMSE)**.  
- Como diagnosticar **viés e variância**.  
- Como comparar o modelo com um **baseline** simples.

---

## Conceitos Envolvidos

### Árvore de Decisão
Um modelo preditivo que divide o espaço de decisão em regiões menores, com base em perguntas (condições) sobre as variáveis.  
É fácil de interpretar e serve tanto para **classificação** quanto **regressão**.

### Critérios de Divisão
- **Gini**: mede a impureza de um nó (padrão para classificação).  
- **Entropia**: mede a quantidade de informação necessária para classificar corretamente as amostras.

### 🔹 Viés e Variância
- **Viés (bias)**: tendência de simplificar demais (subajuste).  
- **Variância**: sensibilidade exagerada aos dados de treino (superajuste).  
O equilíbrio entre ambos é essencial para um bom modelo.

---

## Tecnologias Utilizadas

- **Python 3.10+**
- **Google Colab**
- **Bibliotecas:**
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

---

## Datasets

1. **Breast Cancer Wisconsin (WBC)**  
   📎 [Link](https://assets.datacamp.com/production/repositories/1796/datasets/0eb6987cb9633e4d6aa6cfd11e00993d2387caa4/wbc.csv)

   Usado para **classificação binária** (diagnóstico de câncer).  
   **Target:** `diagnosis`

2. **Auto MPG**  
   📎 [Link](https://assets.datacamp.com/production/repositories/1796/datasets/3781d588cf7b04b1e376c7e9dda489b3e6c7465b/auto.csv)

   Usado para **regressão** (predição de `mpg` — milhas por galão).  
   A coluna `origin` foi codificada via *One-Hot Encoding*.

---

## Etapas do Projeto

### 1. Árvore de Decisão para Classificação
- Importação do dataset *WBC*.
- Divisão em treino e teste.
- Treinamento com `DecisionTreeClassifier`.
- Comparação entre critérios `gini` e `entropy`.
- Avaliação da acurácia (~93%).

### 2. Árvore de Decisão para Regressão
- Importação do dataset *Auto MPG*.
- Codificação das variáveis categóricas (`origin`).
- Treinamento com `DecisionTreeRegressor`.
- Cálculo de métricas de erro (RMSE) para:
  - Treino (`RMSE_train`)
  - Teste (`RMSE_test`)
  - Validação cruzada (`RMSE_CV`)
  - Baseline (modelo que prevê sempre a média)

### 3. Análise de Viés e Variância
- Comparação entre erros de treino, teste e baseline.
- Interpretação dos resultados em relação a *overfitting* e *underfitting*.

### 4. Visualização
- Gráfico de barras mostrando `RMSE_baseline`, `RMSE_train` e `RMSE_test`.

---

## Resultados Obtidos

| Métrica | Valor (aprox.) | Interpretação |
|----------|----------------|---------------|
| **RMSE_train** | 3.85 | Erro médio do modelo no treino |
| **RMSE_test** | 4.82 | Erro médio do modelo no teste |
| **RMSE_baseline** | 8.03 | Erro se o modelo previsse apenas a média |

**Conclusões:**
- O modelo reduziu o erro em quase **40%** em relação ao baseline.  
- Pequena diferença entre treino e teste → **modelo bem equilibrado**.  
- O modelo **aprendeu padrões reais** sem superajustar.

---

## Interpretação Final

O modelo de regressão por árvore de decisão:
- Desempenha **significativamente melhor** do que o baseline.
- Mostra **boa generalização** (sem overfitting).
- Está **adequadamente configurado** com `max_depth=8` e `min_samples_leaf=0.13`.

**Conclusão geral:**  
Os modelos baseados em árvore são eficazes tanto para classificação quanto para regressão, oferecendo bom desempenho e interpretabilidade, mesmo com configurações simples.

---

## Visualização de RMSE

```python
import matplotlib.pyplot as plt

rmse_values = [rmse_baseline, RMSE_train, RMSE_test]
labels = ['Baseline', 'Treino', 'Teste']

plt.bar(labels, rmse_values, color=['gray', 'skyblue', 'orange'])
plt.title('Comparação de RMSE')
plt.ylabel('RMSE (quanto menor, melhor)')
plt.show()
```

---

## Autoria

Projeto desenvolvido por **Ana Meira**  
Assunto: *Aprendizado de Máquina / Modelos Baseados em Árvores*  
Ferramenta: *Google Colab*
