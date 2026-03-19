# Análise de Vendas em E-commerce (Wish)

---

## Visão Geral

Projeto de análise de dados com foco em identificar os fatores que influenciam o sucesso de produtos em um marketplace. A abordagem cobre todo o pipeline analítico: da exploração dos dados até a modelagem preditiva e interpretação de resultados.

---

## Objetivo de Negócio

Responder quais fatores aumentam a probabilidade de sucesso de um produto, apoiando decisões como:

- Precificação estratégica  
- Uso de anúncios (*ad boost*)  
- Aplicação de badges  
- Otimização de catálogo  
- Estratégias de visibilidade  

---

## Dataset

- **Fonte:** Kaggle  
- **Contexto:** Produtos de verão vendidos na Wish  
- **Variáveis utilizadas:** título, preço, preço de varejo, moeda do comprador, unidades vendidas, usa impulsão de anúncios, avaliação, contagem de avaliações, contagem de selos, selo de qualidade do produto, selo de envio rápido, etiquetas (tags), cor do produto, ID do tamanho da variação do produto, estoque da variação do produto, o envio é expresso, países para os quais envia, estoque total, possui banner de urgência, país de origem, contagem de avaliações do vendedor, avaliação do vendedor.

---

## Etapas do Projeto

### 1. Data Cleaning e Preparação

- Tratamento de valores ausentes  
- Ajuste de tipos de dados  
- Remoção de inconsistências  
- Seleção de variáveis relevantes  

---

### 2. Análise Exploratória (EDA)

Principais análises realizadas:

- Distribuição do volume de vendas  
- Relação entre preço e desempenho  
- Impacto de avaliações e ratings  
- Influência de badges no sucesso  
- Análise de frete (shipping)  
- Exploração de tags (WordCloud)  

---

### 3. Feature Engineering

- Criação da variável alvo (sucesso do produto)  
- Transformação de variáveis categóricas  
- Preparação dos dados para modelagem  

---

### 4. Modelagem 

- Algoritmo: Random Forest  
- Divisão treino/teste  
- Treinamento e validação do modelo  

---

## Avaliação do Modelo

- **Accuracy:** 75%  

### Precision
- Classe 0: 0.76  
- Classe 1: 0.75  

### Recall
- Classe 0: 0.78  
- Classe 1: 0.72  

### F1-score
- Classe 0: 0.77  
- Classe 1: 0.73  

---

### Matriz de Confusão

|                | Predito 0 | Predito 1 |
|----------------|----------|----------|
| Real 0         | 196      | 54       |
| Real 1         | 63       | 159      |

---

## Interpretação dos Resultados

- Modelo com desempenho equilibrado entre as classes  
- Boa identificação de produtos de baixo desempenho  
- Perda moderada de produtos com potencial (falsos negativos)  
- Presença de previsões otimistas (falsos positivos)  

---

## Interpretabilidade (SHAP)

A análise com SHAP identificou as variáveis mais relevantes:

- Avaliações dos usuários  
- Quantidade de badges  
- Uso de anúncios (*ad boost*)  
- Preço e competitividade  

Esses fatores explicam diretamente as decisões do modelo.

---

## Principais Insights de Negócio

Os resultados indicam que o sucesso dos produtos é determinado pela combinação de múltiplos fatores, e não por uma variável isolada.

## Principais achados:

Prova social é decisiva: produtos com melhores avaliações e maior volume de reviews apresentam maior número de vendas

Precificação influencia comportamento: descontos percebidos (diferença entre preço real e varejo) impactam diretamente a conversão

Visibilidade ajuda, mas não garante sucesso: ad boost melhora performance, porém não substitui qualidade

Confiança é um fator-chave: badges (qualidade e envio rápido) aumentam a probabilidade de venda

Fatores secundários também contribuem: tags, alcance geográfico e logística têm impacto, mas menor isoladamente

---

## Conclusão:
Produtos com boa reputação, percepção de valor e sinais de confiabilidade tendem a apresentar desempenho superior no marketplace.

---

## Tecnologias Utilizadas

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- SHAP  

---

## Estrutura do Projeto

├── analise-de-vendas-wish.ipynb
├── README.md
├── requirements.txt
├── summer-products-with-rating-and-performance_2020-08

---

## Como Executar

1. Clone o repositório

2. Instale as dependências:
```bash
pip install -r requirements.txt

---

