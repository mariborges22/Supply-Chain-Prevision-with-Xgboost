# Projeto de Análise e Modelagem de Movimentação de Estoque

Este projeto tem como objetivo principal **analisar e modelar os padrões de movimentação de estoque de produtos para otimizar a gestão logística e o planejamento**. Inicialmente, o foco foi a previsão pontual da quantidade movimentada. No entanto, após a avaliação dos resultados, o objetivo evoluiu para **identificar e agrupar padrões de movimentação de estoque** para diferentes combinações de dimensões (Ano, Filial, Local de Estoque, Tipo de Movimentação, Categoria, Subcategoria) utilizando técnicas de clusterização.

Essa redefinição permite uma abordagem mais estratégica, focando na segmentação do estoque em grupos com comportamentos distintos para:

*   Desenvolver estratégias de gestão de estoque mais direcionadas e eficazes.
*   Priorizar a atenção e os recursos para os grupos de maior volume ou variabilidade.
*   Obter insights sobre os padrões de consumo e reposição em diferentes partes da operação.

## Estrutura do Projeto

O projeto está organizado nas seguintes etapas principais, refletidas na estrutura do notebook:

1.  **Montagem do Google Drive e Configuração do Ambiente:** Configuração inicial para acesso aos dados (embora o dataset original não esteja disponível publicamente) e importação das bibliotecas necessárias (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `optuna`, `shap`, `python-calamine`).
2.  **Pré-processamento dos Dados:** Etapa crucial que envolveu a limpeza, tratamento de formatos inconsistentes, remoção de valores ausentes e transformação de variáveis. Os dados históricos de movimentação de estoque foram agregados pela granularidade desejada (ano, filial, local, tipo, categoria, subcategoria) e variáveis categóricas foram codificadas (One-Hot Encoding) para prepará-los para análise e modelagem.
3.  **Modelagem Preditiva (XGBoost):** Construção e avaliação de um modelo XGBoost para prever a quantidade movimentada de estoque (inicialmente na escala transformada).
4.  **Avaliação do Modelo e Análise de Erros:** Análise detalhada do desempenho do modelo, identificação das features mais importantes (utilizando SHAP) e investigação dos erros de previsão em diferentes granularidades.
5.  **Ajuste de Hiperparâmetros com Optuna:** Otimização dos hiperparâmetros do modelo XGBoost para melhorar sua performance preditiva.
6.  **Análise de Métricas na Escala Original e Rejustificativa do Problema:** As previsões do modelo otimizado foram revertidas para a escala original (`qte_movimentacao` utilizando `np.expm1`). A avaliação das métricas (MAE, RMSE, R², MAPE, MedAE) na escala original revelou desafios significativos na previsão pontual, especialmente para grandes volumes. **Esta análise levou à redefinição do problema de negócio, focando em clusterização em vez de previsão pontual.**
7.  **Clusterização:** Aplicação de técnicas de agrupamento (a ser detalhado nas próximas etapas do notebook) para identificar padrões de movimentação de estoque e segmentar os dados agregados em clusters com características similares.
8.  **Análise dos Clusters e Interpretação dos Resultados:** Investigação das características de cada cluster identificado, buscando insights sobre os diferentes comportamentos de movimentação de estoque para informar decisões estratégicas.

## Modelagem Preditiva com XGBoost

### Escolha do Modelo

A escolha do XGBoost (Extreme Gradient Boosting) para a tarefa inicial de previsão de movimentação de estoque foi baseada em suas comprovadas capacidades e eficiência para dados tabulares:

*   **Alta performance em dados tabulares:** XGBoost é amplamente reconhecido por sua precisão em problemas de regressão com dados estruturados como o de movimentação de estoque.
*   **Eficiência computacional:** O algoritmo é otimizado para velocidade e uso eficiente de memória, o que é vantajoso ao trabalhar com grandes volumes de dados históricos.
*   **Capacidade de generalização:** Possui mecanismos internos de regularização (L1 e L2) que ajudam a evitar overfitting, garantindo que o modelo generalize bem para dados não vistos.
*   **Flexibilidade:** Adapta-se bem a variáveis categóricas codificadas (como as obtidas pelo One-Hot Encoding) e pode capturar interações complexas entre features.

### Transformação da Variável Alvo

A variável alvo, `qte_movimentacao`, apresentava uma distribuição assimétrica e incluía valores zero e negativos (representando saídas de estoque). Para lidar com essa característica e melhorar o desempenho do modelo, aplicou-se a transformação `np.log1p(abs(qte_movimentacao)) * np.sign(qte_movimentacao)`. Esta transformação logarítmica (aplicada ao valor absoluto) ajuda a normalizar a distribuição e reduzir a influência de grandes volumes, enquanto a multiplicação pelo sinal original preserva a distinção entre entradas (positivas) e saídas (negativas).

### Ajuste de Hiperparâmetros com Optuna

A performance de um modelo XGBoost é significativamente influenciada por seus hiperparâmetros. Para otimizar a precisão e a capacidade de generalização do nosso modelo para a previsão de movimentação de estoque, utilizamos a biblioteca **Optuna** para automatizar o processo de ajuste de hiperparâmetros.

O Optuna é um otimizador de hiperparâmetros que utiliza estratégias de busca eficientes para encontrar a combinação ideal de configurações que minimiza uma métrica objetivo definida. Neste projeto, o processo de ajuste envolveu:

*   **Definição da Função Objetivo:** Uma função foi criada para treinar e avaliar um modelo XGBoost com um conjunto específico de hiperparâmetros sugeridos pelo Optuna.
*   **Validação Cruzada:** Para obter uma estimativa robusta do desempenho do modelo, utilizamos validação cruzada K-Fold (com 3 splits) dentro da função objetivo. Isso permitiu avaliar o modelo em diferentes subconjuntos dos dados de treino agregados.
*   **Métrica de Otimização:** A métrica utilizada para guiar o Optuna foi o **RMSE (Root Mean Squared Error) médio** obtido na validação cruzada, calculado na escala transformada da variável alvo. O objetivo do Optuna foi minimizar este RMSE.
*   **Espaço de Busca:** O Optuna explorou um espaço de diferentes valores para hiperparâmetros cruciais do XGBoost, incluindo `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha` e `reg_lambda`.
*   **Execução:** O Optuna executou múltiplos "trials" (30 trials definidos), onde em cada trial um novo conjunto de hiperparâmetros era testado e avaliado.

Ao final do processo, o Optuna identificou o conjunto de hiperparâmetros que resultou no menor RMSE médio na validação cruzada, fornecendo a configuração ideal para treinar o modelo final.

### Resultados e Discussão (Escala Transformada)

Após o treinamento do modelo XGBoost com hiperparâmetros otimizados (encontrados via Optuna e validação cruzada) nos dados agregados e com a variável alvo transformada, as seguintes métricas de avaliação foram obtidas no conjunto de teste:

*   **MAE (Erro Absoluto Médio):** 1.22
*   **RMSE (Raiz do Erro Quadrático Médio):** 1.55
*   **R² (Coeficiente de Determinação):** 0.8622
*   **MAPE (Erro Percentual Absoluto Médio):** 49.85%
*   **MedAE (Erro Mediano Absoluto):** 1.02

**Discussão:**

*   **R² de 0.8622:** Este valor indica que o modelo é capaz de explicar aproximadamente 86.22% da variabilidade na variável alvo transformada (`qte_movimentacao_transformed`). Um R² alto na escala transformada sugere que as features selecionadas e o modelo XGBoost capturam a maior parte da relação entre as dimensões e a quantidade movimentada agregada nesta escala.
*   **MAE (1.22) e RMSE (1.55):** Estas métricas representam o erro médio e a raiz do erro quadrático médio na escala transformada. A pequena diferença entre MAE e RMSE (1.22 vs 1.55) sugere que a distribuição dos erros na escala transformada é relativamente centrada, embora o RMSE ligeiramente maior indique a presença de alguns erros maiores que são penalizados mais pesadamente.
*   **MAPE (49.85%) na Escala Transformada:** Este valor indica que, em média, as previsões na escala transformada desviam em quase 50% dos valores reais transformados. É crucial reiterar que este MAPE não pode ser interpretado diretamente como um erro percentual na quantidade real movimentada. Ele é mais sensível a valores próximos de zero na escala transformada (que correspondem a volumes menores na escala original).
*   **MedAE (1.02):** Sendo mais robusto a outliers, o MedAE de 1.02 na escala transformada indica que metade dos erros absolutos estão abaixo deste valor. A proximidade entre MAE e MedAE reforça que a distribuição dos erros na escala transformada não é extremamente assimétrica, mas a diferença sugere a existência de alguns erros maiores.

Em suma, na escala transformada, o modelo apresentou um bom ajuste geral aos dados de teste, capturando a maior parte da variabilidade.

### Análise de Erros e Importância das Features (SHAP)

A análise dos erros na escala transformada, agrupados pelas dimensões categóricas (filial, local, tipo, categoria, subcategoria), revelou que algumas combinações específicas apresentavam erros médios maiores. Isso sugere variabilidade intrínseca ou dados limitados para certas categorias.

A análise de importância das features utilizando SHAP confirmou que as dimensões categóricas, especialmente **tipo de movimentação (SAIDA)**, **locais/filiais específicos** e a **categoria/subcategoria EPI**, são os fatores mais influentes na previsão da quantidade movimentada transformada. Isso valida a relevância dessas features e a abordagem de agregação.

### Resultados e Discussão (Escala Original)

O passo crítico para a interpretação de negócio é reverter as previsões para a escala original (`qte_movimentacao`) utilizando a transformação inversa (`np.expm1`). As métricas na escala original foram:

*   **MAE (Erro Absoluto Médio):** 6528.80
*   **RMSE (Raiz do Erro Quadrático Médio):** 193433.04
*   **R² (Coeficiente de Determinação):** 0.0719
*   **MAPE (Erro Percentual Absoluto Médio):** 360.33%
*   **MedAE (Erro Mediano Absoluto):** 24.33

**Discussão:**

*   **Alto MAE (6528.80) e RMSE (193433.04):** Estes valores indicam que, em média, o erro nas previsões da quantidade real movimentada é significativamente alto. O RMSE, sendo muito maior que o MAE, reforça a presença de erros de previsão consideravelmente grandes para algumas instâncias no conjunto de teste.
*   **Baixo R² (0.0719):** Na escala original, o modelo explica apenas cerca de 7.19% da variabilidade na quantidade real movimentada. Isso contrasta fortemente com o alto R² na escala transformada e evidencia a dificuldade do modelo em prever com precisão a magnitude exata das movimentações quando os dados são revertidos. A transformação, embora útil para o treinamento, amplifica os erros na reversão, especialmente para previsões de grandes volumes.
*   **Altíssimo MAPE (360.33%) e MedAE (24.33):** O MAPE extremamente alto na escala original é um forte indicativo da dificuldade do modelo em lidar com a magnitude dos valores. O MedAE, sendo consideravelmente menor que o MAE, confirma que a distribuição dos erros na escala original é altamente assimétrica, com muitos erros pequenos e alguns erros muito grandes, que distorcem a média (MAE) e o RMSE. Isso aponta para a dificuldade do modelo em prever com precisão os volumes mais extremos de movimentação.

**Conclusão da Modelagem Preditiva:**

Apesar do bom desempenho na escala transformada e da validação da importância das features, as métricas na escala original demonstram que a previsão pontual da quantidade exata de movimentação de estoque, especialmente para grandes volumes, apresenta um erro substancial. Isso limita a aplicabilidade prática da previsão pontual para o planejamento logístico ótimo, que exige maior acurácia na magnitude.

**Este resultado levou à reorientação do projeto para a clusterização.** Identificar e agrupar padrões de movimentação (altos volumes vs. baixos volumes, alta variabilidade vs. baixa variabilidade, etc.) é uma alternativa mais robusta e útil para informar estratégias de gestão de estoque, mesmo sem a capacidade de prever o volume exato com alta precisão. A clusterização permitirá segmentar o estoque em grupos manejáveis e aplicar abordagens de gestão específicas para cada grupo.

## Dataset

O dataset original utilizado neste projeto contém informações confidenciais da empresa e de seus clientes, incluindo detalhes sobre movimentações de estoque, filiais, locais de armazenamento, categorias e subcategorias de produtos, além das quantidades movimentadas. **Portanto, o dataset original não está disponível publicamente neste repositório.**

O notebook demonstra as etapas de pré-processamento, modelagem e análise utilizando um dataset simulado ou uma representação genérica para ilustrar o fluxo de trabalho.

## Tecnologias Utilizadas

*   Python
*   Bibliotecas Python: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `optuna`, `shap`, `python-calamine` (para leitura de arquivos .xlsx).

## Como Utilizar o Notebook

1.  **Clonar o Repositório:** Clone este repositório para sua máquina local ou Google Drive.
2.  **Abrir no Google Colab:** Abra o notebook (.ipynb) no Google Colab.
3.  **Montar o Google Drive:** Execute a célula para montar seu Google Drive.
4.  **Substituir o Dataset:** **Para executar o notebook, você precisará substituir o arquivo de dados original (`bi_movimentacao.xlsx`) por um dataset com estrutura similar, contendo as colunas relevantes para a análise (Ano da movimentação, Filial, Local de estoque, Tipo de movimentação, Categoria do produto, Subcategoria do produto, Quantidade movimentada).**
5.  **Executar as Células:** Prossiga executando as células do notebook sequencialmente. Certifique-se de ter as bibliotecas necessárias instaladas (as células iniciais do notebook devem lidar com isso).

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues para relatar bugs ou sugerir melhorias, ou enviar pull requests com novas funcionalidades ou correções.



---

**Nota:** Este README fornece uma visão geral do projeto e suas etapas. Detalhes específicos sobre os dados originais utilizados, a empresa ou resultados confidenciais foram omitidos por questões de privacidade e segurança. O notebook serve como um template demonstrando a metodologia aplicada.
