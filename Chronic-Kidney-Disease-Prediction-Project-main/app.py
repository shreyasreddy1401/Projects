from flask import Flask, render_template, request, url_for
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = pickle.load(open('Kidney.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        sg = float(request.form['sg'])
        htn = float(request.form['htn'])
        hemo = float(request.form['hemo'])
        dm = float(request.form['dm'])
        al = float(request.form['al'])
        appet = float(request.form['appet'])
        rc = float(request.form['rc'])
        pc = float(request.form['pc'])

        values = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
        prediction = model.predict(values)

        # Determine symptoms
        symptoms = []
        if sg < 1.010:
            symptoms.append(("Low Specific Gravity", "Possible kidney dysfunction"))
        if htn == 1:
            symptoms.append(("Hypertension", "High blood pressure"))
        if hemo < 12:
            symptoms.append(("Low Hemoglobin", "Anemia"))
        if dm == 1:
            symptoms.append(("Diabetes Mellitus", "Elevated blood sugar levels"))
        if al > 0:
            symptoms.append(("Albumin in urine", "Possible kidney damage"))
        if appet == 0:
            symptoms.append(("Poor Appetite", "Potential signs of chronic kidney disease"))
        if rc < 4.5:
            symptoms.append(("Low Red Blood Cell Count", "Anemia"))
        if pc == 1:
            symptoms.append(("Abnormal Pus Cells", "Possible infection or inflammation"))

        # Generate the comparison graph
        create_comparison_graph(sg, hemo, rc, al, appet)

        return render_template('result.html', prediction=prediction, symptoms=symptoms)

def create_comparison_graph(sg, hemo, rc, al, appet):
    user_values = {
        'Specific Gravity': sg,
        'Hemoglobin': hemo,
        'Red Blood Cell Count': rc,
        'Albumin': al,
        'Appetite': appet
    }

    healthy_values = {
        'Specific Gravity': 1.020,
        'Hemoglobin': 13.5,
        'Red Blood Cell Count': 5.0,
        'Albumin': 0,
        'Appetite': 1
    }

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    fig.suptitle('Comparison of User Input Values and Healthy Values')

    for idx, (param, user_val) in enumerate(user_values.items()):
        ax = axs[idx // 2, idx % 2]
        healthy_val = healthy_values[param]
        ax.bar(['User', 'Healthy'], [user_val, healthy_val], color=['blue', 'green'])
        ax.set_title(param)
        ax.set_ylim(0, max(user_val, healthy_val) + 1)

    fig.delaxes(axs[2, 1])
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    graph_path = os.path.join('static', 'comparison_graph.png')
    plt.savefig(graph_path)
    plt.close()

if __name__ == "__main__":
    app.run(debug=True)
