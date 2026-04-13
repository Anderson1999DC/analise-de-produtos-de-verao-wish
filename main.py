from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

modelo = joblib.load("modelo_wish_success.pkl")
colunas = joblib.load("colunas_wish.pkl")

app = FastAPI(
    title="API Análise de Produtos Wish",
    description="Modelo Random Forest para prever sucesso de produtos na plataforma Wish",
    version="1.0.0"
)

class Produto(BaseModel):
    price: float
    retail_price: float
    uses_ad_boosts: int
    rating: float
    badges_count: int
    badge_product_quality: int
    badge_fast_shipping: int
    product_variation_inventory: int
    shipping_is_express: int
    countries_shipped_to: int
    inventory_total: int
    has_urgency_banner: int
    merchant_rating: float
    tags_count: int

@app.get("/")
def root():
    return {"status": "online", "modelo": "Random Forest", "versao": "1.0.0"}

@app.get("/app", response_class=HTMLResponse)
def interface():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict(produto: Produto):
    dados = produto.dict()
    dados["discount"] = dados["retail_price"] - dados["price"]

    df = pd.DataFrame([dados])
    df_final = df.reindex(columns=colunas, fill_value=0)

    predicao = modelo.predict(df_final)[0]
    probabilidade = modelo.predict_proba(df_final)[0]

    return {
        "sucesso": int(predicao),
        "resultado": "Produto com alto potencial de sucesso" if predicao == 1 else "Produto com baixo potencial de sucesso",
        "probabilidade_sucesso": round(float(probabilidade[1]), 4),
        "probabilidade_insucesso": round(float(probabilidade[0]), 4),
        "modelo": "RandomForestClassifier"
    }