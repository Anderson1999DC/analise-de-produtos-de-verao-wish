# Análise de Vendas de Produtos de Verão Wish

### EDA · Classificação · Random Forest · SHAP · GridSearchCV · FastAPI · Docker · Deploy

&nbsp;

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![SHAP](https://img.shields.io/badge/SHAP-Explicabilidade-FF6B6B?style=for-the-badge)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/API-online-28a745?style=for-the-badge)](https://api-wish-ml.onrender.com)

&nbsp;
> Análise exploratória e modelo preditivo para identificar os fatores que determinam
> o sucesso comercial de produtos de verão na plataforma Wish com explicabilidade via SHAP
> e deploy em produção com API REST containerizada.

&nbsp;

**[Acessar interface interativa](https://api-wish-ml.onrender.com/app)** &nbsp;|&nbsp; **[Documentação da API](https://api-wish-ml.onrender.com/docs)**

---

## Índice

- [Contexto](#contexto)
- [Objetivos](#objetivos)
- [Perguntas de Negócio](#perguntas-de-negócio)
- [Pipeline do Projeto](#pipeline-do-projeto)
- [Tecnologias](#tecnologias-utilizadas)
- [Dataset](#dataset)
- [Análise Exploratória](#análise-exploratória)
- [Machine Learning](#machine-learning)
- [Principais Resultados](#principais-resultados)
- [API em Produção](#api-em-produção)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Autor](#autor)

---

## Contexto

Projeto de análise de dados e Machine Learning aplicado ao e-commerce, utilizando dados reais de produtos de verão vendidos na plataforma Wish em 2020. O objetivo é entender quais fatores determinam o sucesso comercial de um produto no marketplace, combinando análise exploratória orientada por hipóteses com Random Forest e explicabilidade via SHAP. O modelo foi colocado em produção como API REST containerizada.

| Etapa | Descrição |
|---|---|
| **EDA orientada** | Validação de hipóteses de negócio via análise visual |
| **Feature Engineering** | Criação de `discount`, `tags_count` e `income` |
| **Modelagem** | Random Forest otimizado via GridSearchCV |
| **Explicabilidade** | SHAP para entender o impacto de cada variável |
| **Deploy** | API REST com FastAPI + Docker + Render |

---

## Objetivos

- Identificar os fatores que mais influenciam o sucesso de produtos na Wish
- Responder perguntas de negócio com análise exploratória orientada por hipóteses
- Construir um modelo preditivo de classificação (sucesso vs. sem sucesso)
- Usar SHAP para explicar as previsões individuais do modelo
- Criar uma API REST com FastAPI e containerizar com Docker
- Fazer deploy em produção com link público acessível

---

## Perguntas de Negócio

As seguintes hipóteses foram investigadas durante a análise:

| # | Hipótese |
|---|---|
| 1 | Produtos com maior desconto percebido vendem mais? |
| 2 | Ad boosts aumentam as vendas? |
| 3 | Avaliações melhores aumentam vendas? |
| 4 | Badges de qualidade e envio rápido importam? |
| 5 | Quantidade de tags auxilia vendas? |
| 6 | Quais tags estão associadas a produtos de sucesso? |
| 7 | Produtos com mais países de entrega vendem mais? |

---

## Pipeline do Projeto

```mermaid
flowchart TD
    A([Dataset\nWish Summer\n2020]) --> B[Limpeza\nNulos · Seleção de colunas]
    B --> C[EDA orientada\nValidação de hipóteses]
    C --> D[Feature Engineering\ndiscount · tags_count · income]
    D --> E[Definição de sucesso\nfaturamento > 7.000 USD]
    E --> F[Random Forest\nGridSearchCV · 5 folds]
    F --> G[SHAP\nExplicabilidade do modelo]
    G --> H[API REST\nFastAPI · Docker]
    H --> I([Deploy\nRender · Link público])

    style A fill:#4A90D9,color:#fff,stroke:none
    style I fill:#28a745,color:#fff,stroke:none
    style B fill:#6C757D,color:#fff,stroke:none
    style C fill:#6C757D,color:#fff,stroke:none
    style D fill:#6C757D,color:#fff,stroke:none
    style E fill:#6C757D,color:#fff,stroke:none
    style F fill:#6C757D,color:#fff,stroke:none
    style G fill:#6C757D,color:#fff,stroke:none
    style H fill:#6C757D,color:#fff,stroke:none
```

---

## Tecnologias Utilizadas

| Tecnologia | Uso no Projeto |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Linguagem principal |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Manipulação e análise dos dados |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Operações numéricas |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | Random Forest e GridSearchCV |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white) | Visualizações e gráficos |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white) | Histogramas e análises comparativas |
| ![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=flat-square) | Explicabilidade do modelo |
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) | API REST para servir o modelo em produção |
| ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) | Containerização da aplicação |
| ![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white) | Hospedagem do deploy em produção |

---

## Dataset

**Fonte:** [Summer Products and Sales in E-Commerce Wish](https://www.kaggle.com/datasets/jmmvutu/summer-products-and-sales-in-ecommerce-wish) Kaggle

| Característica | Detalhe |
|---|---|
| Volume | ~1.573 produtos |
| Período | Agosto de 2020 |
| Variável target | `success` — faturamento > $7.000 |
| Features selecionadas | 22 colunas originais |
| Features criadas | `discount`, `tags_count`, `income` |

---

## Análise Exploratória

### Avaliações e Sucesso

![Distribuição de Avaliação por Sucesso](assets/distribuicao_avaliacao_por_sucesso.png)

> Produtos com melhores avaliações tendem consistentemente a vender mais **prova social é o principal driver de conversão na Wish**.

### Quantidade de Tags e Alcance

![Distribuição de Tags por Sucesso](assets/distribuicao_quantidade_de_tags.png)

> Produtos com mais tags têm maior discoverabilidade na plataforma o algoritmo de busca da Wish favorece produtos mais tagueados.

### Tags dos Produtos de Sucesso

![Tags de Sucesso](assets/tags_sucesso.png)

> Wordcloud comparativo: os termos mais frequentes nos produtos campeões de venda vs. produtos sem sucesso revela as categorias e nichos com maior demanda.

### Análise de Logística

![Shipping Analysis](assets/shipping_analysis.png)

> Produtos com cobertura em mais países de entrega apresentam correlação com maiores volumes de venda alcance global amplia o mercado potencial.

---

## Machine Learning

### Features Mais Importantes

![Feature Importance](assets/feature_importance_wish.png)

> Rating e merchant_rating lideram em importância confirmando que reputação e prova social são os fatores mais determinantes para o sucesso de um produto na Wish.

### Explicabilidade com SHAP

![SHAP Summary Plot](assets/SHAP.png)

> O SHAP mostra não apenas **quais** features importam, mas **como** cada valor influencia a previsão. Pontos vermelhos à direita aumentam a probabilidade de sucesso; à esquerda, diminuem.

### Otimização com GridSearchCV

| Parâmetro | Valores Testados |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_features` | 2, 4, 6, 8 |
| `bootstrap` | True, False |

---

## Principais Resultados

**Fatores que mais determinam o sucesso de um produto na Wish:**

| # | Fator | Insight |
|---|---|---|
| 1 | **Rating e volume de avaliações** | Prova social é o principal driver de conversão |
| 2 | **Desconto percebido** | Diferença entre retail_price e price influencia a conversão |
| 3 | **Quantidade de tags** | Mais tags = maior discoverabilidade orgânica |
| 4 | **Cobertura geográfica** | Mais países atendidos = maior mercado potencial |
| 5 | **Ad boosts** | Efeito positivo mas não determinante — qualidade supera publicidade |
| 6 | **Badges** | Contribuem para confiança mas não garantem vendas isoladamente |

**Aplicações do modelo:**
- Score de probabilidade de sucesso para novos produtos antes do lançamento
- Apoio a vendedores na otimização de listagens
- Identificação de nichos e tags estratégicas por categoria

---

## API em Produção

### Interface Interativa

[![Interface do Modelo](assets/interface.png)](https://api-wish-ml.onrender.com/app)

> Acesse a interface em: **[api-wish-ml.onrender.com/app](https://api-wish-ml.onrender.com/app)**

### Documentação Swagger

[![Swagger UI](assets/Swagger.png)](https://api-wish-ml.onrender.com/docs)

> Documentação completa da API em: **[api-wish-ml.onrender.com/docs](https://api-wish-ml.onrender.com/docs)**

### Exemplo de Requisição

```bash
curl -X POST https://api-wish-ml.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "price": 15.0,
    "retail_price": 25.0,
    "uses_ad_boosts": 0,
    "rating": 4.2,
    "badges_count": 1,
    "badge_product_quality": 1,
    "badge_fast_shipping": 0,
    "product_variation_inventory": 10,
    "shipping_is_express": 0,
    "countries_shipped_to": 40,
    "inventory_total": 50,
    "has_urgency_banner": 0,
    "merchant_rating": 4.0,
    "tags_count": 12
  }'
```

### Resposta

```json
{
  "sucesso": 1,
  "resultado": "Produto com alto potencial de sucesso",
  "probabilidade_sucesso": 0.7832,
  "probabilidade_insucesso": 0.2168,
  "modelo": "RandomForestClassifier"
}
```

### Endpoints disponíveis

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/` | Status da API |
| `GET` | `/app` | Interface interativa |
| `GET` | `/docs` | Documentação Swagger |
| `POST` | `/predict` | Análise de potencial do produto |

---

## Estrutura do Repositório

```
analise-de-produtos-de-verao-wish/
│
├──  assets/                                  # Gráficos e imagens
│   ├── distribuicao_avaliacao_por_sucesso.png
│   ├── distribuicao_quantidade_de_tags.png
│   ├── feature_importance_wish.png
│   ├── SHAP.png
│   ├── shipping_analysis.png
│   ├── tags_sucesso.png
│   ├── interface.png
│   └── Swagger.png
│
├──  analise_de_vendas_wis.ipynb              # Notebook completo
├──  main.py                                  # API FastAPI
├──  index.html                               # Interface interativa
├──  Dockerfile                               # Containerização
├──  modelo_wish_success.pkl                  # Modelo Random Forest treinado
├──  colunas_wish.pkl                         # Features esperadas pela API
├──  summer-products-with-rating-and-performance_2020-08.csv
├──  requirements.txt                         # Dependências do projeto
└──  README.md                                # Documentação do projeto
```

---

## Autor

<div align="center">

<img src="https://github.com/Anderson1999DC.png" width="100px" style="border-radius:50%"/>

**Anderson Coelho**
*Cientista de Dados*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anderson-coelho-42671634a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Anderson1999DC)

</div>

---

<div align="center">
