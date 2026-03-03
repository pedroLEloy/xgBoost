"""
🏦 SISTEMA DE ANÁLISE DE SOLVÊNCIA - COM SUPORTE A VALORES NEGATIVOS E CUTOFF DINÂMICO
Formata números automaticamente (incluindo negativos)
Gráficos se ajustam automaticamente ao cutoff definido
"""

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
    print("✅ SHAP disponível para análise de importância")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP não instalado. Para análise detalhada: pip install shap")

# ============================================================
# 📊 CONFIGURAÇÕES DO MODELO
# ============================================================

MODEL_PATH = r"C:\Users\9000306\Desktop\projeto_solvencia_pj\results\modelo_otimizado_balanced.json"

def verificar_modelo(path):
    """
    Tenta carregar o modelo e exibe MsgBox caso falhe,
    orientando o usuário a verificar a VPN.
    """
    import tkinter as tk
    from tkinter import messagebox

    try:
        m = xgb.XGBClassifier()
        m.load_model(path)
        print("✅ Modelo carregado com sucesso!")

        # Teste rápido
        test_data = np.zeros((1, 35))
        try:
            test_pred = m.predict_proba(test_data)
            print(f"✅ Modelo testado: {test_pred[0]}")
        except Exception as e:
            print(f"⚠️ Aviso no teste do modelo: {e}")

        return m

    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")

        # Exibir MsgBox com tkinter
        try:
            root = tk.Tk()
            root.withdraw()          # Oculta a janela principal
            root.attributes("-topmost", True)  # MsgBox sempre no topo
            messagebox.showerror(
                title="❌ Erro ao carregar o modelo",
                message=(
                    f"Não foi possível carregar o modelo preditivo.\n\n"
                    f"📂 Caminho esperado:\n{path}\n\n"
                    f"🔒 Verifique se você está conectado à VPN corporativa "
                    f"e se a unidade de rede 'L:\\' está mapeada.\n\n"
                    f"Detalhes do erro:\n{e}"
                )
            )
            root.destroy()
        except Exception as tk_err:
            print(f"⚠️ Não foi possível exibir MsgBox: {tk_err}")


model = verificar_modelo(MODEL_PATH)

FEATURES = [
    # ✅ Features históricas de inadimplência (2022-2023)
    "Inadim_Qtd_Meses_12m_2022", "Inadim_Qtd_Meses_12m_2023",
    
    # ✅ Indicadores financeiros estruturais
    "Liquidez_Geral", "Liquidez_Corrente", "Liquidez_Seca", "Liquidez_Imediata",
    "Margem_Bruta", "Margem_Liquida", "Margem_EBITDA", "ROA", "ROE",
    "EBITDA / Despesas Financeiras",
    "Composição de Endividamento (CE)", "Índice de Endividamento Geral (IEG)",
    "Dívida Líquida / Patrimônio Líquido", "Dívida Líquida / Faturamento",
    "Patrimônio/ Faturamento", "Participação do Capital Bancário", "Recursos não correntes",
    "Imobilização do Patrimônio Líquido (IPL)", "Empréstimos e Financiamentos/AT (Fin. AT)",
    "Participação de Empréstimos na Dívida", "Financiamentos CP/AC",
    "Grau de Alavancagem Financeira (GAF)", "DL/ EBITDA",
    "Participação de Capital de Terceiros (PCT)", "Capital Circulante Líquido",
    "NCG / FAT", "Contas a receber / Fat.", "Fornecedores / NCG",
    "Ciclo_de_Financeiro", "Ciclo_Operacional",
    "Prazo_Medio_Recebimento", "Prazo_Medio_Estoque", "Prazo_Medio_Fornecedores"
]

# ============================================================
# 🔧 FUNÇÕES DE FORMATAÇÃO COM SUPORTE A NEGATIVOS
# ============================================================

def formatar_input_decimal(valor):
    if not valor or valor == "":
        return ""
    
    is_negative = str(valor).strip().startswith('-')
    valor_limpo = re.sub(r'[^\d]', '', str(valor))
    
    if not valor_limpo:
        return "-" if is_negative else ""
    
    valor_limpo = valor_limpo.lstrip('0') or '0'
    num_digitos = len(valor_limpo)
    
    if num_digitos == 1:
        inteiro = "0"
        decimal = "0" + valor_limpo
    elif num_digitos == 2:
        inteiro = "0"
        decimal = valor_limpo
    else:
        inteiro = valor_limpo[:-2]
        decimal = valor_limpo[-2:]
    
    inteiro_formatado = ""
    for i, digito in enumerate(reversed(inteiro)):
        if i > 0 and i % 3 == 0:
            inteiro_formatado = '.' + inteiro_formatado
        inteiro_formatado = digito + inteiro_formatado
    
    resultado = f"{inteiro_formatado},{decimal}"
    return f"-{resultado}" if is_negative else resultado

def formatar_input_percentual(valor):
    return formatar_input_decimal(valor)

def formatar_input_moeda(valor):
    return formatar_input_decimal(valor)

def formatar_input_inteiro(valor):
    if not valor or valor == "":
        return ""
    
    is_negative = str(valor).strip().startswith('-')
    valor_limpo = re.sub(r'[^\d]', '', str(valor))
    
    if not valor_limpo:
        return "-" if is_negative else ""
    
    valor_limpo = valor_limpo.lstrip('0') or '0'
    
    inteiro_formatado = ""
    for i, digito in enumerate(reversed(valor_limpo)):
        if i > 0 and i % 3 == 0:
            inteiro_formatado = '.' + inteiro_formatado
        inteiro_formatado = digito + inteiro_formatado
    
    return f"-{inteiro_formatado}" if is_negative else inteiro_formatado

def desformatar_numero(valor_str):
    if not valor_str or str(valor_str).strip() == "":
        return 0.0
    
    valor_str = str(valor_str)
    is_negative = valor_str.strip().startswith('-')
    valor_str = valor_str.replace('.', '').replace(',', '.').replace('-', '')
    
    try:
        numero = float(valor_str)
        return -numero if is_negative else numero
    except:
        return 0.0

# ============================================================
# 🎨 FUNÇÃO DE PREDIÇÃO
# ============================================================

def prever_solvencia(*args):
    print(f"🔍 prever_solvencia chamado com {len(args)} argumentos")
    
    if model is None:
        error_msg = "❌ ERRO: Modelo não carregado"
        print(error_msg)
        return error_msg, "Verifique o caminho do modelo", None, None
    
    try:
        valores = list(args[:-1])
        cutoff = args[-1]
        
        print(f"📊 Processando {len(valores)} valores, cutoff={cutoff}%")
        
        inadim_2022 = int(desformatar_numero(valores[0])) if valores[0] else 0
        inadim_2023 = int(desformatar_numero(valores[1])) if valores[1] else 0
        
        valores_desformatados = [
            inadim_2022,
            inadim_2023,
            desformatar_numero(valores[2]),
            desformatar_numero(valores[3]),
            desformatar_numero(valores[4]),
            desformatar_numero(valores[5]),
            desformatar_numero(valores[6]) / 100,
            desformatar_numero(valores[7]) / 100,
            desformatar_numero(valores[8]) / 100,
            desformatar_numero(valores[9]) / 100,
            desformatar_numero(valores[10]) / 100,
            desformatar_numero(valores[11]) / 100,
            desformatar_numero(valores[12]) / 100,
            desformatar_numero(valores[13]) / 100,
            desformatar_numero(valores[14]) / 100,
            desformatar_numero(valores[15]) / 100,
            desformatar_numero(valores[16]) / 100,
            desformatar_numero(valores[17]) / 100,
            desformatar_numero(valores[18]) / 100,
            desformatar_numero(valores[19]) / 100,
            desformatar_numero(valores[20]) / 100,
            desformatar_numero(valores[21]) / 100,
            desformatar_numero(valores[22]) / 100,
            desformatar_numero(valores[23]) / 100,
            desformatar_numero(valores[24]) / 100,
            desformatar_numero(valores[25]) / 100,
            desformatar_numero(valores[26]) / 100,
            desformatar_numero(valores[27]) / 100,
            desformatar_numero(valores[28]) / 100,
            desformatar_numero(valores[29]) / 100,
            int(desformatar_numero(valores[30])) if valores[30] else 0,
            int(desformatar_numero(valores[31])) if valores[31] else 0,
            int(desformatar_numero(valores[32])) if valores[32] else 0,
            int(desformatar_numero(valores[33])) if valores[33] else 0,
            int(desformatar_numero(valores[34])) if valores[34] else 0,
        ]
        
        data = dict(zip(FEATURES, valores_desformatados))
        df = pd.DataFrame([data])
        df = df[FEATURES]
        
        proba = model.predict_proba(df)[0]
        prob_insolvente = proba[0]
        prob_solvente = proba[1]
        
        cutoff_decimal = cutoff / 100
        previsao = "INSOLVENTE ⚠️" if prob_solvente < cutoff_decimal else "SOLVENTE ✅"
        
        margem_muito_bom = cutoff_decimal + 0.15
        margem_bom = cutoff_decimal + 0.05
        margem_medio = cutoff_decimal - 0.10
        
        if prob_solvente >= margem_muito_bom:
            nivel_risco = "[MUITO BOM]"
        elif prob_solvente >= margem_bom:
            nivel_risco = "[BOM]"
        elif prob_solvente >= margem_medio:
            nivel_risco = "[MÉDIO]"
        else:
            nivel_risco = "[RUIM]"
        
        resultado = f"""
        ## 🎯 RESULTADO DA ANÁLISE
        
        ### Classificação: **{previsao}**
        
        ### 📊 Probabilidades:
        - **Probabilidade de Insolvência:** {prob_insolvente*100:.2f}%
        - **Probabilidade de Solvência:** {prob_solvente*100:.2f}%
        
        ### ⚠️ Nível de Risco: **{nivel_risco}**
        
        ### ⚙️ Cutoff: {cutoff:.0f}% | Recall: {recall_0}% | Precision: {precision_0}%
        """
        
        if previsao == "INSOLVENTE ⚠️":
            interpretacao = f"""
            ### 🚨 EMPRESA CLASSIFICADA COMO INSOLVENTE
            
            **Recomendações:**
            1. ✅ Análise detalhada dos demonstrativos
            2. ✅ Verificar garantias e histórico
            3. ✅ Considerar reduzir limite de crédito
            4. ✅ Aumentar monitoramento
            
            ⚠️ Precision {precision_0}% - pode ser falso alarme. Confirmar manualmente se necessário.
            """
        else:
            interpretacao = f"""
            ### ✅ EMPRESA CLASSIFICADA COMO SOLVENTE
            
            **Risco de inadimplência:** {prob_insolvente*100:.1f}%
            
            **Recomendações:**
            - Manter monitoramento periódico
            - Avaliar limite conforme política interna
            
            ℹ️ Recall {recall_0}% - alguns insolventes podem não ser detectados.
            """
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        ax1.bar(['Solvência'], [prob_solvente*100], color=['#44ff44'])
        ax1.set_ylabel('Probabilidade (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Probabilidade de Solvência', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.text(0, prob_solvente*100 + 2, f'{prob_solvente*100:.1f}%', 
                ha='center', fontweight='bold', fontsize=12)
        ax1.axhline(y=cutoff, color='red', linestyle='--', linewidth=2, label=f'Cutoff: {cutoff:.0f}%')
        ax1.text(0.5, cutoff, f'  Cutoff: {cutoff:.0f}%', va='center', fontweight='bold', 
                color='red', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ruim_max = margem_medio * 100
        medio_max = cutoff
        bom_max = margem_bom * 100
        
        ax1.axhspan(0, ruim_max, alpha=0.1, color='red', label='Ruim')
        ax1.axhspan(ruim_max, medio_max, alpha=0.1, color='orange', label='Médio')
        ax1.axhspan(medio_max, bom_max, alpha=0.1, color='lightgreen', label='Bom')
        ax1.axhspan(bom_max, 100, alpha=0.1, color='green', label='Muito Bom')
        ax1.legend(loc='upper right', fontsize=9)
        
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 1)
        ax2.barh(0.5, ruim_max, left=0, height=0.3, color='#ff4444', alpha=0.7)
        ax2.barh(0.5, medio_max - ruim_max, left=ruim_max, height=0.3, color='#FFD700', alpha=0.7)
        ax2.barh(0.5, bom_max - medio_max, left=medio_max, height=0.3, color='#90EE90', alpha=0.7)
        ax2.barh(0.5, 100 - bom_max, left=bom_max, height=0.3, color='#44ff44', alpha=0.7)
        
        if ruim_max > 5:
            ax2.text(ruim_max/2, 0.5, 'Ruim', ha='center', va='center', fontsize=10, fontweight='bold')
        if (medio_max - ruim_max) > 5:
            ax2.text((ruim_max + medio_max)/2, 0.5, 'Médio', ha='center', va='center', fontsize=10, fontweight='bold')
        if (bom_max - medio_max) > 5:
            ax2.text((medio_max + bom_max)/2, 0.5, 'Bom', ha='center', va='center', fontsize=9, fontweight='bold')
        if (100 - bom_max) > 5:
            ax2.text((bom_max + 100)/2, 0.5, 'Muito bom', ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax2.axvline(x=cutoff, color='red', linestyle='--', linewidth=2, alpha=0.8)
        cutoff_label_x = cutoff + 3 if cutoff < 80 else cutoff - 3
        cutoff_label_ha = 'left' if cutoff < 80 else 'right'
        ax2.text(cutoff_label_x, 0.85, f'Cutoff\n{cutoff:.0f}%', ha=cutoff_label_ha, fontsize=9, 
                fontweight='bold', color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        risco_pos = prob_solvente * 100
        ax2.scatter(risco_pos, 0.5, s=200, marker='v', color='black', zorder=5)
        ax2.text(risco_pos, 0.15, f'{risco_pos:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        ax2.set_yticks([])
        ax2.set_xlabel('Probabilidade de Solvência (%)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Nível de Risco: {nivel_risco}', fontsize=14, fontweight='bold')
        
        ticks_dinamicos = sorted([0, ruim_max, cutoff, bom_max, 100])
        ticks_filtrados = []
        for i, tick in enumerate(ticks_dinamicos):
            if i == 0 or abs(tick - ticks_filtrados[-1]) > 5:
                ticks_filtrados.append(tick)
        ax2.set_xticks(ticks_filtrados)
        
        plt.tight_layout()
        
        fig_shap = None
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df)
                
                fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
                shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
                
                feature_names = FEATURES
                values = shap_vals[0]
                
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': values,
                    'Abs SHAP': np.abs(values)
                })
                shap_df = shap_df.nlargest(15, 'Abs SHAP')
                
                colors_shap = ['#ff4444' if x < 0 else '#44ff44' for x in shap_df['SHAP Value']]
                ax_shap.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors_shap)
                ax_shap.set_xlabel('Impacto na Predição (SHAP Value)', fontsize=12, fontweight='bold')
                ax_shap.set_title('Top 15 Features Mais Importantes na Decisão', fontsize=14, fontweight='bold')
                ax_shap.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax_shap.grid(True, alpha=0.3, axis='x')
                
                for i, (idx, row) in enumerate(shap_df.iterrows()):
                    value = row['SHAP Value']
                    ax_shap.text(value, i, f' {value:.3f}', 
                               va='center', ha='left' if value > 0 else 'right',
                               fontweight='bold', fontsize=9)
                
                plt.tight_layout()
                print("✅ Gráfico SHAP gerado com sucesso!")
                
            except Exception as e:
                print(f"⚠️ Erro ao gerar gráfico SHAP: {e}")
                fig_shap = None
        
        relatorio = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "classificacao": previsao,
            "prob_insolvente": float(prob_insolvente),
            "prob_solvente": float(prob_solvente),
            "nivel_risco": nivel_risco,
            "cutoff": cutoff
        }
        
        relatorio_json = json.dumps(relatorio, indent=2, ensure_ascii=False)
        
        print("✅ Predição concluída com sucesso!")
        return resultado, interpretacao, fig, fig_shap, relatorio_json
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ ERRO na predição: {str(e)}")
        print(error_trace)
        return f"❌ ERRO: {str(e)}", f"```\n{error_trace}\n```", None, None, None

# ============================================================
# 🎨 INTERFACE PRINCIPAL (SEM LOGIN)
# ============================================================

custom_css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Análise de Solvência Agro") as app:
    
    gr.Markdown("# 🏦 SISTEMA DE ANÁLISE DE SOLVÊNCIA AGRO created by: Pedro Eloy", elem_id="title")
    
    with gr.Row():
        with gr.Column(scale=2):
        
            gr.Markdown("## 🏢 Identificação")
            liable = gr.Textbox(label="Liable", value="", placeholder="Ex: 10000123")

            gr.Markdown("---")

            gr.Markdown("## 📅 Histórico de Inadimplência (meses)")
            with gr.Row():
                inadim_2022 = gr.Textbox(label="Inadim 2022 (0-12)", value="", placeholder="Ex: 1,73")
                inadim_2023 = gr.Textbox(label="Inadim 2023 (0-12)", value="", placeholder="Ex: 4,53")
            
            gr.Markdown("Nestes campos são as quantidades de meses que o cliente ficou inadimplente.")
            
            gr.Markdown("## 💧 Liquidez")
            with gr.Row():
                liq_geral = gr.Textbox(label="Liquidez Geral", value="", placeholder="Ex: 1,14")
                liq_corrente = gr.Textbox(label="Liquidez Corrente", value="", placeholder="Ex: 1,26")
            with gr.Row():
                liq_seca = gr.Textbox(label="Liquidez Seca", value="", placeholder="Ex: 0,83")
                liq_imediata = gr.Textbox(label="Liquidez Imediata", value="", placeholder="Ex: 0,07")
            
            gr.Markdown("---")
            
            gr.Markdown("## 📈 Rentabilidade (%)")
            with gr.Row():
                mg_bruta = gr.Textbox(label="Margem Bruta", value="", placeholder="Ex: 13,58")
                mg_liquida = gr.Textbox(label="Margem Líquida", value="", placeholder="Ex: 2,16")
                mg_ebitda = gr.Textbox(label="Margem EBITDA", value="", placeholder="Ex: 2,97")
            with gr.Row():
                roa = gr.Textbox(label="ROA", value="", placeholder="Ex: 2,60")
                roe = gr.Textbox(label="ROE", value="", placeholder="Ex: 9,41")
            
            gr.Markdown("---")
            
            gr.Markdown("## 🛡️ Cobertura (%)")
            ebitda_desp = gr.Textbox(label="EBITDA / Desp. Financeiras", value="", placeholder="Ex: 119,51")
            
            gr.Markdown("---")
            
            gr.Markdown("## 💰 Endividamento (%)")
            with gr.Row():
                comp_endiv = gr.Textbox(label="Comp. Endividamento (CE)", value="", placeholder="Ex: 60,05")
                ieg = gr.Textbox(label="IEG", value="", placeholder="Ex: 81,85")
            with gr.Row():
                div_pl = gr.Textbox(label="Dív. Líq. / PL", value="", placeholder="Ex: 49,06")
                div_fat = gr.Textbox(label="Dív. Líq. / Fat.", value="", placeholder="Ex: 7,98")
            
            gr.Markdown("---")
            
            gr.Markdown("## 🏗️ Estrutura de Capital (%)")
            with gr.Row():
                pat_fat = gr.Textbox(label="Patrim. / Fat.", value="", placeholder="Ex: 7,33")
                cap_banc = gr.Textbox(label="Cap. Bancário", value="", placeholder="Ex: 80,73")
            with gr.Row():
                rec_nc = gr.Textbox(label="Rec. Não Correntes", value="", placeholder="Ex: 42,97")
                ipl = gr.Textbox(label="IPL", value="", placeholder="Ex: 36,61")
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
    
    # EVENTOS DE FORMATAÇÃO AUTOMÁTICA
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

# ============================================================
# 🚀 INICIALIZAÇÃO
# ============================================================

if __name__ == "__main__":
    print("🚀 Iniciando sistema...")
    print("💡 Ajuste o slider de cutoff e os gráficos se adaptarão automaticamente!")
    
    app.launch(
        share=False, 
        server_name="127.0.0.1", 
        server_port=7860, 
        show_error=True
    )