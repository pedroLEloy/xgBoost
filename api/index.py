import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime
import re

recall_0 = 69.70
precision_0 = 77.31

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelo_otimizado_balanced.json")

def verificar_modelo(path):
    try:
        m = xgb.XGBClassifier()
        m.load_model(path)
        test_data = np.zeros((1, 35))
        test_pred = m.predict_proba(test_data)
        return m
    except Exception as e:
        raise Exception(f"Erro ao carregar modelo: {e}")

modelo = verificar_modelo(MODEL_PATH)

FEATURE_NAMES = [
    'Inadimplência 2022', 'Inadimplência 2023',
    'Liquidez Geral', 'Liquidez Corrente', 'Liquidez Seca', 'Liquidez Imediata',
    'Margem Bruta (%)', 'Margem Líquida (%)', 'Margem EBITDA (%)', 'ROA (%)', 'ROE (%)',
    'EBITDA / Desp. Financeiras',
    'Composição do Endividamento (%)', 'IEG (%)', 'Dívida / PL (%)', 'Dívida / Faturamento (%)',
    'Patrimônio / Faturamento (%)', 'Cap. Bancário / Dívida (%)', 'Rec. Não Corrente / AT (%)', 'IPL (%)', 'Emp. e Fin (Fin. / AT) (%)', 'Part. Empr. na Dívida (%)', 'Fin. CP / AC (%)',
    'GAF', 'DL / EBITDA', 'PCT (%)', 'Cap. Circ. Líq.',
    'NCG / FAT (%)', 'Contas Rec. / Fat. (%)', 'Fornec. / NCG (%)',
    'Ciclo Financeiro (dias)', 'Ciclo Operacional (dias)', 'PMR (dias)', 'PME (dias)', 'PMP (dias)'
]

def limpar_numero(valor_str):
    if not valor_str or valor_str.strip() == "":
        return 0.0
    valor_str = str(valor_str).strip()
    valor_str = valor_str.replace("R$", "").replace("%", "").strip()
    valor_str = valor_str.replace(".", "").replace(",", ".")
    try:
        return float(valor_str)
    except:
        return 0.0

def formatar_input_decimal(valor):
    num = limpar_numero(valor)
    return f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def formatar_input_percentual(valor):
    num = limpar_numero(valor)
    return f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + "%"

def formatar_input_moeda(valor):
    num = limpar_numero(valor)
    return f"R$ {num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def formatar_input_inteiro(valor):
    num = limpar_numero(valor)
    return str(int(num))

def prever_solvencia(inadim_2022, inadim_2023, liq_geral, liq_corrente, liq_seca, liq_imediata,
                     mg_bruta, mg_liquida, mg_ebitda, roa, roe, ebitda_desp,
                     comp_endiv, ieg, div_pl, div_fat, pat_fat, cap_banc, rec_nc, ipl, fin_at, emp_div, fin_cp,
                     gaf, dl_ebitda, pct, ccl, ncg_fat, ct_rec, forn_ncg,
                     ciclo_fin, ciclo_oper, pmr, pme, pmf, cutoff):
    
    valores = [
        inadim_2022, inadim_2023,
        liq_geral, liq_corrente, liq_seca, liq_imediata,
        mg_bruta, mg_liquida, mg_ebitda, roa, roe, ebitda_desp,
        comp_endiv, ieg, div_pl, div_fat, pat_fat, cap_banc, rec_nc, ipl, fin_at, emp_div, fin_cp,
        gaf, dl_ebitda, pct, ccl, ncg_fat, ct_rec, forn_ncg,
        ciclo_fin, ciclo_oper, pmr, pme, pmf
    ]
    
    valores_limpos = [limpar_numero(v) for v in valores]
    df_input = pd.DataFrame([valores_limpos], columns=FEATURE_NAMES)
    
    proba = modelo.predict_proba(df_input)[0]
    prob_insolvente = proba[0] * 100
    prob_solvente = proba[1] * 100
    
    cutoff_decimal = cutoff / 100.0
    classe_pred = 0 if prob_insolvente >= (cutoff * 1.0) else 1
    
    resultado_md = f"""
### 🎯 Resultado da Análise
- **Probabilidade de Insolvência:** {prob_insolvente:.2f}%
- **Probabilidade de Solvência:** {prob_solvente:.2f}%
- **Classificação (cutoff {cutoff}%):** {'🔴 INSOLVENTE' if classe_pred == 0 else '🟢 SOLVENTE'}
"""
    
    interp_md = f"""
### 📊 Interpretação
Com base no modelo treinado:
- Recall para Insolventes: {recall_0:.2f}%
- Precisão para Insolventes: {precision_0:.2f}%
"""
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Insolvente', 'Solvente'],
        y=[prob_insolvente, prob_solvente],
        marker_color=['red', 'green'],
        text=[f'{prob_insolvente:.2f}%', f'{prob_solvente:.2f}%'],
        textposition='auto'
    ))
    fig.update_layout(
        title=f'Probabilidades (Cutoff: {cutoff}%)',
        yaxis_title='Probabilidade (%)',
        showlegend=False
    )
    
    shap_plot = None
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(df_input)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
            else:
                shap_vals = shap_values[0]
            
            indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
            top_features = [FEATURE_NAMES[i] for i in indices]
            top_values = [shap_vals[i] for i in indices]
            
            shap_fig = go.Figure()
            shap_fig.add_trace(go.Bar(
                y=top_features,
                x=top_values,
                orientation='h',
                marker_color=['red' if v < 0 else 'green' for v in top_values]
            ))
            shap_fig.update_layout(
                title='Top 10 Features (SHAP)',
                xaxis_title='SHAP Value',
                yaxis_title='Feature'
            )
            shap_plot = shap_fig
        except:
            pass
    
    resultado_json = {
        "timestamp": datetime.now().isoformat(),
        "probabilidades": {
            "insolvente": round(prob_insolvente, 2),
            "solvente": round(prob_solvente, 2)
        },
        "classificacao": "INSOLVENTE" if classe_pred == 0 else "SOLVENTE",
        "cutoff": cutoff,
        "metricas_modelo": {
            "recall_insolventes": recall_0,
            "precisao_insolventes": precision_0
        }
    }
    
    json_output = json.dumps(resultado_json, indent=2, ensure_ascii=False)
    
    return resultado_md, interp_md, fig, shap_plot, json_output

with gr.Blocks(title="🏦 Análise de Solvência", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🏦 Sistema de Análise de Solvência")
    gr.Markdown("Preencha os indicadores financeiros abaixo para análise.")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📋 Dados de Entrada")
            
            gr.Markdown("### 🚨 Inadimplência")
            with gr.Row():
                inadim_2022 = gr.Textbox(label="Inadimplência 2022", value="", placeholder="Ex: 0 ou 1")
                inadim_2023 = gr.Textbox(label="Inadimplência 2023", value="", placeholder="Ex: 0 ou 1")
            
            gr.Markdown("---")
            gr.Markdown("## 💧 Liquidez")
            with gr.Row():
                liq_geral = gr.Textbox(label="Liquidez Geral", value="", placeholder="Ex: 1,23")
                liq_corrente = gr.Textbox(label="Liquidez Corrente", value="", placeholder="Ex: 1,45")
            with gr.Row():
                liq_seca = gr.Textbox(label="Liquidez Seca", value="", placeholder="Ex: 0,98")
                liq_imediata = gr.Textbox(label="Liquidez Imediata", value="", placeholder="Ex: 0,12")
            
            gr.Markdown("---")
            gr.Markdown("## 📈 Rentabilidade (%)")
            with gr.Row():
                mg_bruta = gr.Textbox(label="Margem Bruta", value="", placeholder="Ex: 25,50")
                mg_liquida = gr.Textbox(label="Margem Líquida", value="", placeholder="Ex: 8,30")
            with gr.Row():
                mg_ebitda = gr.Textbox(label="Margem EBITDA", value="", placeholder="Ex: 15,20")
                roa = gr.Textbox(label="ROA", value="", placeholder="Ex: 5,60")
            with gr.Row():
                roe = gr.Textbox(label="ROE", value="", placeholder="Ex: 12,40")
                ebitda_desp = gr.Textbox(label="EBITDA / Desp. Fin.", value="", placeholder="Ex: 3,20")
            
            gr.Markdown("---")
            gr.Markdown("## 💰 Endividamento (%)")
            with gr.Row():
                comp_endiv = gr.Textbox(label="Comp. Endividamento", value="", placeholder="Ex: 45,30")
                ieg = gr.Textbox(label="IEG", value="", placeholder="Ex: 55,20")
            with gr.Row():
                div_pl = gr.Textbox(label="Dívida / PL", value="", placeholder="Ex: 120,50")
                div_fat = gr.Textbox(label="Dívida / Fat.", value="", placeholder="Ex: 35,80")
            with gr.Row():
                pat_fat = gr.Textbox(label="Patrimônio / Fat.", value="", placeholder="Ex: 28,90")
                cap_banc = gr.Textbox(label="Cap. Banc. / Dívida", value="", placeholder="Ex: 65,40")
            with gr.Row():
                rec_nc = gr.Textbox(label="Rec. NC / AT", value="", placeholder="Ex: 18,70")
                ipl = gr.Textbox(label="IPL", value="", placeholder="Ex: 85,30")
            with gr.Row():
                fin_at = gr.Textbox(label="Emp. e Fin (Fin. / AT)", value="", placeholder="Ex: 13,00")
                emp_div = gr.Textbox(label="Part. Empr. na Dívida", value="", placeholder="Ex: 24,26")
            fin_cp = gr.Textbox(label="Fin. CP / AC", value="", placeholder="Ex: 10,41")
            
            gr.Markdown("---")
            gr.Markdown("## ⚖️ Alavancagem (%)")
            with gr.Row():
                gaf = gr.Textbox(label="GAF", value="", placeholder="Ex: 104,63")
                dl_ebitda = gr.Textbox(label="DL / EBITDA", value="", placeholder="Ex: 149,32")
            with gr.Row():
                pct = gr.Textbox(label="PCT", value="", placeholder="Ex: 393,48")
                ccl = gr.Textbox(label="Cap. Circ. Líq.", value="", placeholder="Ex: 12,19")
            
            gr.Markdown("---")
            gr.Markdown("## 🔄 Ciclo Operacional (%)")
            with gr.Row():
                ncg_fat = gr.Textbox(label="NCG / FAT", value="", placeholder="Ex: 38,87")
                ct_rec = gr.Textbox(label="Contas Rec. / Fat.", value="", placeholder="Ex: 53,40")
            forn_ncg = gr.Textbox(label="Fornec. / NCG", value="", placeholder="Ex: 352,04")
            
            gr.Markdown("---")
            gr.Markdown("## 📅 Prazos (dias)")
            with gr.Row():
                ciclo_fin = gr.Textbox(label="Ciclo Financeiro", value="", placeholder="Ex: 126")
                ciclo_oper = gr.Textbox(label="Ciclo Operacional", value="", placeholder="Ex: 303")
            with gr.Row():
                pmr = gr.Textbox(label="PMR", value="", placeholder="Ex: 146")
                pme = gr.Textbox(label="PME", value="", placeholder="Ex: 130")
            pmf = gr.Textbox(label="PMP", value="", placeholder="Ex: 92")
            
            gr.Markdown("---")
            gr.Markdown("## ⚙️ Cutoff")
            cutoff = gr.Slider(0, 100, value=50, step=1, label="Cutoff (%)")
            gr.Markdown("⚠️ **Ajuste o cutoff para redefinir as faixas de risco dinamicamente nos gráficos**")
            
            btn = gr.Button("🔍 ANALISAR", variant="primary", size="lg")
        
        with gr.Column(scale=3):
            gr.Markdown("## 📊 Resultados")
            resultado_md = gr.Markdown()
            interp_md = gr.Markdown()
            grafico_plot = gr.Plot()
            gr.Markdown("## 🔍 Análise de Importância (SHAP)")
            shap_plot = gr.Plot()
            gr.Markdown("## 📄 JSON")
            json_code = gr.Code(language="json")
    
    liq_geral.change(fn=formatar_input_decimal, inputs=liq_geral, outputs=liq_geral)
    liq_corrente.change(fn=formatar_input_decimal, inputs=liq_corrente, outputs=liq_corrente)
    liq_seca.change(fn=formatar_input_decimal, inputs=liq_seca, outputs=liq_seca)
    liq_imediata.change(fn=formatar_input_decimal, inputs=liq_imediata, outputs=liq_imediata)
    
    mg_bruta.change(fn=formatar_input_percentual, inputs=mg_bruta, outputs=mg_bruta)
    mg_liquida.change(fn=formatar_input_percentual, inputs=mg_liquida, outputs=mg_liquida)
    mg_ebitda.change(fn=formatar_input_percentual, inputs=mg_ebitda, outputs=mg_ebitda)
    roa.change(fn=formatar_input_percentual, inputs=roa, outputs=roa)
    roe.change(fn=formatar_input_percentual, inputs=roe, outputs=roe)
    
    ebitda_desp.change(fn=formatar_input_decimal, inputs=ebitda_desp, outputs=ebitda_desp)
    
    comp_endiv.change(fn=formatar_input_percentual, inputs=comp_endiv, outputs=comp_endiv)
    ieg.change(fn=formatar_input_percentual, inputs=ieg, outputs=ieg)
    div_pl.change(fn=formatar_input_percentual, inputs=div_pl, outputs=div_pl)
    div_fat.change(fn=formatar_input_percentual, inputs=div_fat, outputs=div_fat)
    
    pat_fat.change(fn=formatar_input_percentual, inputs=pat_fat, outputs=pat_fat)
    cap_banc.change(fn=formatar_input_percentual, inputs=cap_banc, outputs=cap_banc)
    rec_nc.change(fn=formatar_input_percentual, inputs=rec_nc, outputs=rec_nc)
    ipl.change(fn=formatar_input_percentual, inputs=ipl, outputs=ipl)
    fin_at.change(fn=formatar_input_percentual, inputs=fin_at, outputs=fin_at)
    emp_div.change(fn=formatar_input_percentual, inputs=emp_div, outputs=emp_div)
    fin_cp.change(fn=formatar_input_percentual, inputs=fin_cp, outputs=fin_cp)
    
    gaf.change(fn=formatar_input_decimal, inputs=gaf, outputs=gaf)
    dl_ebitda.change(fn=formatar_input_decimal, inputs=dl_ebitda, outputs=dl_ebitda)
    pct.change(fn=formatar_input_percentual, inputs=pct, outputs=pct)
    ccl.change(fn=formatar_input_moeda, inputs=ccl, outputs=ccl)
    
    ncg_fat.change(fn=formatar_input_percentual, inputs=ncg_fat, outputs=ncg_fat)
    ct_rec.change(fn=formatar_input_percentual, inputs=ct_rec, outputs=ct_rec)
    forn_ncg.change(fn=formatar_input_percentual, inputs=forn_ncg, outputs=forn_ncg)
    
    inadim_2022.change(fn=formatar_input_inteiro, inputs=inadim_2022, outputs=inadim_2022)
    inadim_2023.change(fn=formatar_input_inteiro, inputs=inadim_2023, outputs=inadim_2023)
    
    ciclo_fin.change(fn=formatar_input_inteiro, inputs=ciclo_fin, outputs=ciclo_fin)
    ciclo_oper.change(fn=formatar_input_inteiro, inputs=ciclo_oper, outputs=ciclo_oper)
    pmr.change(fn=formatar_input_inteiro, inputs=pmr, outputs=pmr)
    pme.change(fn=formatar_input_inteiro, inputs=pme, outputs=pme)
    pmf.change(fn=formatar_input_inteiro, inputs=pmf, outputs=pmf)
    
    btn.click(
        fn=prever_solvencia,
        inputs=[
            inadim_2022, inadim_2023,
            liq_geral, liq_corrente, liq_seca, liq_imediata,
            mg_bruta, mg_liquida, mg_ebitda, roa, roe,
            ebitda_desp,
            comp_endiv, ieg, div_pl, div_fat,
            pat_fat, cap_banc, rec_nc, ipl, fin_at, emp_div, fin_cp,
            gaf, dl_ebitda, pct, ccl,
            ncg_fat, ct_rec, forn_ncg,
            ciclo_fin, ciclo_oper, pmr, pme, pmf,
            cutoff
        ],
        outputs=[resultado_md, interp_md, grafico_plot, shap_plot, json_code]
    )

app.queue()
