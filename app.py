from flask import Flask, render_template, request, send_file, make_response
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)
os.makedirs("exports", exist_ok=True)

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
    
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(79, 70, 229)
        self.cell(0, 10, 'Analyse des Risques Opérationnels R2', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} - Généré le {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(79, 70, 229)
        self.set_fill_color(240, 240, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)
    
    def add_chapter(self, title, content):
        # Vérifier si on a assez d'espace pour ajouter le chapitre
        if self.get_y() > 250 - (15 + (content.count('\n') + 1) * 8):
            self.add_page()
        self.chapter_title(title)
        self.set_font('Arial', size=12)
        self.multi_cell(0, 8, content)
        self.ln(5)
    
    def colored_text(self, text, color=(79, 70, 229)):
        self.set_text_color(*color)
        self.cell(0, 10, text, 0, 1)
        self.set_text_color(0, 0, 0)

def predire_r2(valeurs):
    X = np.array([[i] for i in range(len(valeurs))])
    y = np.array(valeurs)
    model = LinearRegression().fit(X, y)
    pred_2025 = model.predict(np.array([[4], [5], [6], [7]]))
    pred_2026 = model.predict(np.array([[8], [9], [10], [11]]))
    return model.coef_[0], model.intercept_, pred_2025.tolist(), pred_2026.tolist()

def create_plot(labels, values):
    plt.figure(figsize=(10, 6))
    plt.plot(labels, values, marker='o', color='indigo', linestyle='-', linewidth=2)
    plt.fill_between(labels, values, color='indigo', alpha=0.1)
    plt.title('Évolution du Risque Opérationnel R2', fontsize=14)
    plt.xlabel('Périodes', fontsize=12)
    plt.ylabel('Montant (FC)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_pdf(labels, values, historique, commentaire, a, b, pred_2025, pred_2026):
    pdf = PDF()
    pdf.add_page()
    
    # Titre principal
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(79, 70, 229)
    pdf.cell(0, 15, "Rapport d'Analyse des Risques R2", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, "Analyse prédictive des risques opérationnels", 0, 1, 'C')
    pdf.ln(15)
    
    # Section Historique
    pdf.add_chapter("1. Données Historiques", "")
    for label, val in historique:
        pdf.cell(40, 10, label + ":", 0, 0)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{val:,.0f} FC", 0, 1)
        pdf.set_font("Arial", size=12)
    
    # Section Analyse
    pdf.add_chapter("2. Analyse des Résultats", commentaire)
    
    # Section Prédictions
    pred_text = "Prédictions pour 2025:\n"
    for i, val in enumerate(pred_2025):
        pred_text += f"T{i+1} 2025: {val:,.0f} FC\n"
    pred_text += "\nPrédictions pour 2026:\n"
    for i, val in enumerate(pred_2026):
        pred_text += f"T{i+1} 2026: {val:,.0f} FC\n"
    
    pdf.add_chapter("3. Prédictions", pred_text)
    
    # Graphique
    pdf.add_chapter("4. Visualisation Graphique", "")
    img_data = create_plot(labels, values)
    pdf.image(io.BytesIO(base64.b64decode(img_data)), x=10, y=None, w=180)
    
    # Méthodologie
    methodology = f"""
    Les prédictions sont calculées à l'aide d'une régression linéaire simple selon la formule:
    
    y = a * x + b
    
    Où:
    - y = montant prédit
    - x = numéro de période (0 à 3 pour historique, 4-7 pour 2025, 8-11 pour 2026)
    - a = coefficient directeur = {a:,.2f}
    - b = ordonnée à l'origine = {b:,.2f}
    
    Calcul détaillé:
    
    1. Numérotation des périodes:
       - {historique[0][0]} = Période 0
       - {historique[1][0]} = Période 1
       - {historique[2][0]} = Période 2
       - {historique[3][0]} = Période 3
    
    2. Calcul des prédictions:
       - T1 2025 = {a:,.2f} * 4 + {b:,.2f} = {(a*4 + b):,.0f} FC
       - T2 2025 = {a:,.2f} * 5 + {b:,.2f} = {(a*5 + b):,.0f} FC
       - T1 2026 = {a:,.2f} * 8 + {b:,.2f} = {(a*8 + b):,.0f} FC
    
    Cette méthode extrapole la tendance linéaire observée dans les données historiques.
    """
    pdf.add_chapter("5. Méthodologie de Calcul", methodology)
    
    # Conclusion
    conclusion = """
    Ce rapport fournit une analyse prédictive basée sur les données historiques.
    Les résultats doivent être interprétés en tenant compte du contexte opérationnel.
    Pour toute question, veuillez contacter votre responsable des risques.
    """
    pdf.add_chapter("6. Conclusion", conclusion)
    
    # Sauvegarde du PDF
    pdf_path = "exports/rapport_r2.pdf"
    pdf.output(pdf_path)
    return pdf_path

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/formulaire_auto')
def formulaire_auto():
    valeurs = [844938801140, 844938801140, 1261493624000, 1261493624000]
    labels = ["Sept 2023", "Déc 2023", "Mars 2024", "Juin 2024"]
    return render_template("formulaire_auto.html", valeurs=valeurs, labels=labels)

@app.route('/formulaire_custom')
def formulaire_custom():
    return render_template("formulaire_custom.html")

@app.route('/predict', methods=['POST'])
def predict():
    labels = []
    valeurs = []
    for i in range(4):
        label = request.form.get(f'label{i}', f'T{i+1}')
        valeur = request.form.get(f'value{i}', '0')
        labels.append(label)
        valeurs.append(float(valeur))

    a, b, pred_2025, pred_2026 = predire_r2(valeurs)

    last_real = valeurs[-1]
    first_pred = pred_2025[0]
    if first_pred > last_real * 1.1:
        commentaire = "Le risque opérationnel semble en hausse continue pour 2025. Il est conseillé de renforcer les contrôles internes et d'analyser les causes profondes de cette augmentation."
    elif first_pred < last_real * 0.9:
        commentaire = "Une baisse du risque est prévue. Cela peut refléter une amélioration de vos processus de contrôle ou des mesures correctives efficaces."
    else:
        commentaire = "Le risque prévu est stable. Poursuivez vos efforts de maîtrise opérationnelle et surveillez les indicateurs clés."

    all_labels = labels + ["T1 2025", "T2 2025", "T3 2025", "T4 2025", "T1 2026", "T2 2026", "T3 2026", "T4 2026"]
    all_values = valeurs + pred_2025 + pred_2026

    historique_data = list(zip(labels, valeurs))

    pdf_path = generate_pdf(all_labels, all_values, historique_data, commentaire, a, b, pred_2025, pred_2026)

    return render_template("result.html",
                         historique=historique_data,
                         pred2025=pred_2025,
                         pred2026=pred_2026,
                         filepath=pdf_path,
                         commentaire=commentaire,
                         all_labels=all_labels,
                         all_values=all_values)

@app.route('/download')
def download():
    return send_file("exports/rapport_r2.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
