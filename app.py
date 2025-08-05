import pickle
import gradio as gr
import numpy as np
import pandas as pd

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
features = df.columns[:-1]
def add_display_names(features):
    names = {}
    for feature in features:
        if feature == 'creatinine_phosphokinase':
            names[feature] = "Level of the CPK enzyme in the blood (mcg/L)"
        elif feature == 'ejection_fraction':
            names[feature] = "Percentage of blood leaving the heart at each contraction (%)"
        elif feature == 'platelets':
            names[feature] = "Platelets in the blood (kiloplatelets/mL)"
        elif feature == 'serum_creatinine':
            names[feature] = "Level of creatinine in the blood (mg/dL)"
        elif feature == 'serum_sodium':
            names[feature] = "Level of sodium in the blood (mEq/L)"
        elif feature == 'time':
            names[feature] = "Follow-up period (days)"
        else:
            names[feature] = feature.replace('_', ' ').capitalize()
    return names
names = add_display_names(features)

binary_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
binary_labels = {
    "default": ["doesn't have", "has"],
    "smoking": ["doesn't", "does"],
    "sex": ["Female", "Male"]
}

with open("styling.css") as f:
    css = f.read()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(*inputs):
    input_data = np.array(inputs).reshape(1, -1)
    return "They're (most likely) safe" if model.predict(input_data) == 0 else "RIP"

def render_feature(feature):
    if feature in binary_features:
        labels = binary_labels.get(feature if feature in ['smoking', 'sex'] else 'default', binary_labels['default'])
        gr.Markdown(f"### {names[feature]}")
        state = gr.State(0)
        btn = gr.Button(labels[0])
        def toggle(value, labels=labels):
            new_value = 1 - value
            return labels[new_value], new_value
        btn.click(fn=toggle, inputs=state, outputs=[btn, state])
        return btn, state
    else:
        comp = gr.Number(label=names[feature], placeholder="...")
        return comp, comp

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Heart Failure Prediction", elem_classes="title")

    input_components = []
    with gr.Row():
        for col_features in [features[:len(features)//3], features[len(features)//3: 2*len(features)//3], features[2*len(features)//3:]]:
            with gr.Column():
                for feature in col_features:
                    comp, input_ref = render_feature(feature)
                    input_components.append(input_ref)

    predict_btn = gr.Button("Predict")
    output_text = gr.Text(label="Model Prediction")
    predict_btn.click(fn=predict, inputs=input_components, outputs=output_text)

demo.launch()
