
from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
os.makedirs("exports", exist_ok=True)

def predire_r2(valeurs):
    X = np.array([[i] for i in range(len(valeurs))])
    y = np.array(valeurs)
    model = LinearRegression().fit(X, y)
    pred_2025 = model.predict(np.array([[4], [5], [6], [7]]))
    pred_2026 = model.predict(np.array([[8], [9], [10], [11]]))
    return pred_2025.tolist(), pred_2026.tolist()

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

    pred_2025, pred_2026 = predire_r2(valeurs)

    last_real = valeurs[-1]
    first_pred = pred_2025[0]
    if first_pred > last_real * 1.1:
        commentaire = "📈 Le risque opérationnel semble en hausse continue pour 2025. Il est conseillé de renforcer les contrôles internes."
    elif first_pred < last_real * 0.9:
        commentaire = "📉 Une baisse du risque est prévue. Cela peut refléter une amélioration de vos processus de contrôle."
    else:
        commentaire = "🔄 Le risque prévu est stable. Poursuivez vos efforts de maîtrise opérationnelle."

    all_labels = labels + ["T1 2025", "T2 2025", "T3 2025", "T4 2025", "T1 2026", "T2 2026", "T3 2026", "T4 2026"]
    all_values = valeurs + pred_2025 + pred_2026

    df = pd.DataFrame({"Trimestre": all_labels, "R2": all_values})
    filepath = "exports/prediction_r2.xlsx"
    df.to_excel(filepath, index=False)

    return render_template("result.html",
                           historique=zip(labels, valeurs),
                           pred2025=pred_2025,
                           pred2026=pred_2026,
                           filepath=filepath,
                           commentaire=commentaire,
                           all_labels=all_labels,
                           all_values=all_values)

@app.route('/download')
def download():
    return send_file("exports/prediction_r2.xlsx", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
