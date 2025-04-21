from lime.lime_text import LimeTextExplainer
import shap
import numpy as np

def explain_with_lime(model, vectorizer, text_samples):
    explainer = LimeTextExplainer(class_names=['negative', 'positive'])
    exp = explainer.explain_instance(text_samples, model.predict_proba, num_features=10)
    return exp.as_list()

def explain_with_shap(model, vectorizer, text_samples):
    # Assuming the model is a scikit-learn model
    explainer = shap.Explainer(model)
    shap_values = explainer(vectorizer.transform(text_samples))
    return shap_values

def get_shap_summary(shap_values, feature_names):
    return shap.summary_plot(shap_values, feature_names=feature_names)