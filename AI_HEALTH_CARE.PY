from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from groq import Groq
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = "yoursecretkey"   # keep or change

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print("GROQ KEY LOADED:", bool(api_key))  # IMPORTANT

client = Groq(api_key=api_key)

# ------------ LOAD MODELS ------------

# Blood
blood_model = joblib.load("blood_model.pkl")
blood_scaler = joblib.load("blood_scaler.pkl")
blood_feature_names = joblib.load("blood_feature_names.pkl")

# Heart / Diabetes / Kidney
heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")
heart_feature_names = joblib.load("heart_feature_names.pkl")

diabetes_model = joblib.load("diabetes_model.pkl")
diabetes_scaler = joblib.load("diabetes_scaler.pkl")
diabetes_feature_names = joblib.load("diabetes_feature_names.pkl")

kidney_model = joblib.load("kidney_model.pkl")
kidney_scaler = joblib.load("kidney_scaler.pkl")
kidney_feature_names = joblib.load("kidney_feature_names.pkl")

# Text models for symptom analysis (loaded within functions as needed)


# ------------ HELPERS ------------

def build_vector(form, feature_names):
    """Generic helper: build numeric vector for any disease from form."""
    values = []
    any_filled = False
    for feat in feature_names:
        val_str = (form.get(feat, "") or "").strip()
        if val_str != "":
            any_filled = True
        try:
            val = float(val_str) if val_str != "" else 0.0
        except (TypeError, ValueError):
            val = 0.0
        values.append(val)

    if not any_filled:
        return None
    return np.array(values, dtype=float).reshape(1, -1)


def predict_from_text_description(description):
    """Simple text-only prediction for NONE option."""
    if not description:
        return None, None
    _ = text_tfidf.transform([description])
    return text_label, 1.0   # constant label + 100%


def analyze_description_for_disease(description, disease_type):
    """Analyze text description in context of specific disease type."""
    if not description or not description.strip():
        return None, None
    
    # Disease-specific keyword analysis
    desc_lower = description.lower()
    
    if disease_type == "blood":
        # Blood-related symptoms and conditions
        blood_keywords = {
            "anemia": ["tired", "fatigue", "weak", "pale", "dizzy", "short breath"],
            "infection": ["fever", "chills", "sweat", "pain", "sore"],
            "diabetes": ["thirsty", "frequent urine", "hungry", "tired", "blurry vision"],
            "liver disease": ["jaundice", "yellow skin", "abdominal pain", "nausea", "fatigue"],
            "kidney disease": ["swelling", "fatigue", "poor appetite", "trouble concentrating"]
        }
        
        for condition, symptoms in blood_keywords.items():
            if any(symptom in desc_lower for symptom in symptoms):
                return f"Blood-related symptoms suggest possible {condition}", 0.7
                
        return "General blood health concerns mentioned", 0.5
        
    elif disease_type == "heart":
        # Heart-related symptoms
        heart_keywords = {
            "heart disease": ["chest pain", "short breath", "fatigue", "dizzy", "palpitations"],
            "hypertension": ["headache", "dizzy", "nosebleed", "fatigue"],
            "arrhythmia": ["irregular heartbeat", "palpitations", "dizzy", "faint"],
            "heart failure": ["short breath", "swelling", "fatigue", "cough"]
        }
        
        for condition, symptoms in heart_keywords.items():
            if any(symptom in desc_lower for symptom in symptoms):
                return f"Heart-related symptoms suggest possible {condition}", 0.7
                
        return "General cardiovascular concerns mentioned", 0.5
        
    elif disease_type == "diabetes":
        # Diabetes-related symptoms
        diabetes_keywords = {
            "diabetes": ["thirsty", "frequent urine", "hungry", "tired", "blurry vision", "slow healing"],
            "hypoglycemia": ["shaky", "sweaty", "hungry", "dizzy", "irritable"],
            "hyperglycemia": ["thirsty", "frequent urine", "fatigue", "blurry vision"]
        }
        
        for condition, symptoms in diabetes_keywords.items():
            if any(symptom in desc_lower for symptom in symptoms):
                return f"Diabetes-related symptoms suggest possible {condition}", 0.7
                
        return "General metabolic concerns mentioned", 0.5
        
    elif disease_type == "kidney":
        # Kidney-related symptoms
        kidney_keywords = {
            "kidney disease": ["swelling", "fatigue", "poor appetite", "trouble concentrating", "pain"],
            "uti": ["painful urine", "frequent urine", "burning", "fever"],
            "kidney stones": ["severe pain", "blood urine", "nausea", "frequent urine"]
        }
        
        for condition, symptoms in kidney_keywords.items():
            if any(symptom in desc_lower for symptom in symptoms):
                return f"Kidney-related symptoms suggest possible {condition}", 0.7
                
        return "General urinary/renal concerns mentioned", 0.5
    
    return "Description analyzed for general health concerns", 0.4


SYSTEM_PROMPT = (
    "You are a healthcare AI assistant acting like a doctor during consultation.\n"
    "ONLY answer medical and health-related questions.\n\n"

    "CONVERSATION FLOW (MANDATORY):\n"
    "Step 1: When the user asks a health question for the FIRST time,\n"
    "ASK about past medical history BEFORE giving advice.\n\n"

    "Ask these clearly:\n"
    "- Do you have BP (Blood Pressure), Sugar (Diabetes Mellitus)?\n"
    "- Any past surgeries or hospitalizations?\n"
    "- Any long-term medicines?\n"
    "- Any recent medical reports?\n\n"

    "Step 2: AFTER the user replies to history questions:\n"
    "- If they say NO previous issues → give general guidance only\n"
    "- If they mention issues → personalize the response based on that\n\n"

    "FINAL ANSWER FORMAT (MANDATORY):\n"
    "🧾 Possible Condition:\n"
    "- Short, non-alarming condition\n\n"

    "🩺 Explanation:\n"
    "- Simple explanation\n"
    "- Expand medical terms with FULL FORM in brackets\n\n"

    "📊 Normal Values (if applicable):\n"
    "- Mention normal ranges\n\n"

    "💡 What You Can Do Now:\n"
    "- Step-by-step actions\n\n"

    "⚠ When to See a Doctor:\n"
    "- Clear warning signs\n\n"

    "Rules:\n"
    "- Use simple language\n"
    "- Do NOT overdiagnose\n"
    "- If greeting, ask for symptoms\n"
)


# ------------ ROUTES ------------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/assistant", methods=["GET", "POST"])
def assistant():
    """
    Main page: user types description + chooses disease.
    On POST: open correct disease form or do text-only prediction.
    """
    if request.method == "POST":
        description = (request.form.get("description") or "").strip()
        choice = request.form.get("disease_type")

        if choice == "blood":
            return render_template(
                "blood_form.html",
                description=description,
                bloodfeatures=blood_feature_names,
            )

        elif choice == "heart":
            return render_template(
                "heart_form.html",
                description=description,
                heartfeatures=heart_feature_names,
            )

        elif choice == "diabetes":
            return render_template(
                "diabetes_form.html",
                description=description,
                diabetesfeatures=diabetes_feature_names,
            )

        elif choice == "kidney":
            return render_template(
                "kidney_form.html",
                description=description,
                kidneyfeatures=kidney_feature_names,
            )

        elif choice == "none":
            label, score = predict_from_text_description(description)
            if label is None:
                main_prediction = "No sufficient text to predict."
            else:
                main_prediction = f"Text-based category: {label} (confidence {score*100:.1f}%)"
            return render_template(
                "result.html",
                description=description,
                mode="textonly",
                mainprediction=main_prediction,
                probs=None,
            )

        else:
            main_prediction = "Please choose a supported disease type."
            return render_template(
                "result.html",
                description=description,
                mode="error",
                mainprediction=main_prediction,
                probs=None,
            )

    return render_template("assistant.html")


# ---------- BLOOD PREDICT ----------

@app.route("/predict_blood", methods=["POST"])
def predict_blood():
    description = (request.form.get("description") or "").strip()
    vec = build_vector(request.form, blood_feature_names)
    # collect submitted numeric values to show in assistant
    values = {feat: (request.form.get(feat) or "").strip() for feat in blood_feature_names}
    
    # Initialize results
    ml_prediction = None
    text_analysis = None
    combined_probs = []
    
    # Check if numerical values are provided
    if vec is not None:
        vec_scaled = blood_scaler.transform(vec)
        proba = blood_model.predict_proba(vec_scaled)[0]
        classes = blood_model.classes_
        
        ml_probs = sorted(
            [(str(cls), float(p)) for cls, p in zip(classes, proba)],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        
        ml_prediction = f"Blood test analysis: {ml_probs[0][0]}"
        combined_probs.extend(ml_probs)
    
    # Check if description is provided
    if description:
        analysis_result, confidence = analyze_description_for_disease(description, "blood")
        if analysis_result:
            text_analysis = f"Symptom analysis: {analysis_result}"
            combined_probs.append((analysis_result, float(confidence)))
    
    # Ensure all values are JSON serializable
    combined_probs = [(str(name), float(prob)) for name, prob in combined_probs]
    
    # Determine main prediction and response
    if ml_prediction and text_analysis:
        # Both provided - show both
        main_prediction = f"{ml_prediction}\n{text_analysis}"
        mode = "blood_combined"
    elif ml_prediction:
        # Only ML prediction
        main_prediction = ml_prediction
        mode = "blood"
    elif text_analysis:
        # Only text analysis
        main_prediction = text_analysis
        mode = "blood"
        combined_probs = [(analysis_result, confidence)]
    else:
        # Nothing provided
        main_prediction = "Please provide blood test values or a description of your blood-related concerns."
        mode = "blood"
        combined_probs = None
            
    return render_template(
        "result.html",
        description=description,
        mode=mode,
        mainprediction=main_prediction,
        probs=combined_probs,
        ml_prediction=ml_prediction,
        text_analysis=text_analysis,
        values=values,
    )


# ---------- HEART PREDICT ----------

@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    description = (request.form.get("description") or "").strip()
    vec = build_vector(request.form, heart_feature_names)
    # collect submitted numeric values to show in assistant
    values = {feat: (request.form.get(feat) or "").strip() for feat in heart_feature_names}
    # Initialize results
    ml_prediction = None
    text_analysis = None
    combined_probs = []
    
    # Check if numerical values are provided
    if vec is not None:
        vec_scaled = heart_scaler.transform(vec)
        proba = heart_model.predict_proba(vec_scaled)[0]
        classes = heart_model.classes_
        
        ml_probs = sorted(
            [(str(cls), float(p)) for cls, p in zip(classes, proba)],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        
        ml_prediction = f"Heart test analysis: {ml_probs[0][0]}"
        combined_probs.extend(ml_probs)
    
    # Check if description is provided
    if description:
        analysis_result, confidence = analyze_description_for_disease(description, "heart")
        if analysis_result:
            text_analysis = f"Symptom analysis: {analysis_result}"
            combined_probs.append((analysis_result, float(confidence)))
    
    # Ensure all values are JSON serializable
    combined_probs = [(str(name), float(prob)) for name, prob in combined_probs]
    
    # Determine main prediction and response
    if ml_prediction and text_analysis:
        # Both provided - show both
        main_prediction = f"{ml_prediction}\n{text_analysis}"
        mode = "heart_combined"
    elif ml_prediction:
        # Only ML prediction
        main_prediction = ml_prediction
        mode = "heart"
    elif text_analysis:
        # Only text analysis
        main_prediction = text_analysis
        mode = "heart"
        combined_probs = [(analysis_result, confidence)]
    else:
        # Nothing provided
        main_prediction = "Please provide heart test values or a description of your cardiovascular concerns."
        mode = "heart"
        combined_probs = None
            
    return render_template(
        "result.html",
        description=description,
        mode=mode,
        mainprediction=main_prediction,
        probs=combined_probs,
        ml_prediction=ml_prediction,
        text_analysis=text_analysis,
        values=values,
    )



# ---------- DIABETES PREDICT ----------

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    description = (request.form.get("description") or "").strip()
    vec = build_vector(request.form, diabetes_feature_names)
    # collect submitted numeric values to show in assistant
    values = {feat: (request.form.get(feat) or "").strip() for feat in diabetes_feature_names}
    # Initialize results
    ml_prediction = None
    text_analysis = None
    combined_probs = []
    
    # Check if numerical values are provided
    if vec is not None:
        vec_scaled = diabetes_scaler.transform(vec)
        proba = diabetes_model.predict_proba(vec_scaled)[0]
        classes = diabetes_model.classes_
        
        ml_probs = sorted(
            [(str(cls), float(p)) for cls, p in zip(classes, proba)],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        
        ml_prediction = f"Diabetes test analysis: {ml_probs[0][0]}"
        combined_probs.extend(ml_probs)
    
    # Check if description is provided
    if description:
        analysis_result, confidence = analyze_description_for_disease(description, "diabetes")
        if analysis_result:
            text_analysis = f"Symptom analysis: {analysis_result}"
            combined_probs.append((analysis_result, float(confidence)))
    
    # Ensure all values are JSON serializable
    combined_probs = [(str(name), float(prob)) for name, prob in combined_probs]
    
    # Determine main prediction and response
    if ml_prediction and text_analysis:
        # Both provided - show both
        main_prediction = f"{ml_prediction}\n{text_analysis}"
        mode = "diabetes_combined"
    elif ml_prediction:
        # Only ML prediction
        main_prediction = ml_prediction
        mode = "diabetes"
    elif text_analysis:
        # Only text analysis
        main_prediction = text_analysis
        mode = "diabetes"
        combined_probs = [(analysis_result, confidence)]
    else:
        # Nothing provided
        main_prediction = "Please provide diabetes test values or a description of your metabolic concerns."
        mode = "diabetes"
        combined_probs = None

    return render_template(
        "result.html",
        description=description,
        mode=mode,
        mainprediction=main_prediction,
        probs=combined_probs,
        ml_prediction=ml_prediction,
        text_analysis=text_analysis,
        values=values,
    )

# ---------- KIDNEY PREDICT ----------

@app.route("/predict_kidney", methods=["POST"])
def predict_kidney():
    description = (request.form.get("description") or "").strip()
    vec = build_vector(request.form, kidney_feature_names)
    # collect submitted numeric values to show in assistant
    values = {feat: (request.form.get(feat) or "").strip() for feat in kidney_feature_names}
    # Initialize results
    ml_prediction = None
    text_analysis = None
    combined_probs = []
    
    # Check if numerical values are provided
    if vec is not None:
        vec_scaled = kidney_scaler.transform(vec)
        proba = kidney_model.predict_proba(vec_scaled)[0]
        classes = kidney_model.classes_
        
        ml_probs = sorted(
            [(str(cls), float(p)) for cls, p in zip(classes, proba)],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        
        ml_prediction = f"Kidney test analysis: {ml_probs[0][0]}"
        combined_probs.extend(ml_probs)
    
    # Check if description is provided
    if description:
        analysis_result, confidence = analyze_description_for_disease(description, "kidney")
        if analysis_result:
            text_analysis = f"Symptom analysis: {analysis_result}"
            combined_probs.append((analysis_result, float(confidence)))
    
    # Ensure all values are JSON serializable
    combined_probs = [(str(name), float(prob)) for name, prob in combined_probs]
    
    # Determine main prediction and response
    if ml_prediction and text_analysis:
        # Both provided - show both
        main_prediction = f"{ml_prediction}\n{text_analysis}"
        mode = "kidney_combined"
    elif ml_prediction:
        # Only ML prediction
        main_prediction = ml_prediction
        mode = "kidney"
    elif text_analysis:
        # Only text analysis
        main_prediction = text_analysis
        mode = "kidney"
        combined_probs = [(analysis_result, confidence)]
    else:
        # Nothing provided
        main_prediction = "Please provide kidney test values or a description of your urinary/renal concerns."
        mode = "kidney"
        combined_probs = None
            
    return render_template(
        "result.html",
        description=description,
        mode=mode,
        mainprediction=main_prediction,
        probs=combined_probs,
        ml_prediction=ml_prediction,
        text_analysis=text_analysis,
        values=values,
    )


@app.route("/chat-ui")
def chat_ui():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Please describe your health concern."})

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )

        reply = completion.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        print("REAL ERROR:", e)
        return jsonify({"reply": f"ERROR → {str(e)}"})


@app.route("/assistant_direct", methods=["POST"])
def assistant_direct():
    """Return direct advice based on an assessment string.
    This route tells the assistant NOT to ask any follow-up questions and
    to produce actionable suggestions based on the provided assessment.
    """
    try:
        data = request.get_json(force=True)
        assessment_text = data.get("assessment", "").strip()
        disease = data.get("disease", "").strip()
        values = data.get("values", {})

        if not assessment_text:
            return jsonify({"reply": "No assessment provided."}), 400

        # System prompt override: do NOT ask questions; act on the assessment
        direct_prompt = (
            "You are a healthcare AI assistant. You will be given the results of a clinical assessment. "
            "DO NOT ask any follow-up questions. Instead, interpret the assessment, give a concise possible condition, "
            "provide simple explanation, normal ranges (if applicable), immediate actions the user can take, and clear 'when to see a doctor' guidance. "
            "Be concise, avoid overdiagnosis, and present steps the user can follow."
        )

        # If no numeric values were provided, instruct assistant to list recommended tests/values for the disease
        if disease and (not values or all(v=="" for v in values.values())):
            direct_prompt += (
                f" Also, since no numeric lab/test values were provided for {disease}, include a concise list of the most relevant lab tests or measurements to collect for {disease} and their typical normal ranges. Do not ask the user to provide them — just list the tests and ranges."
            )

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": direct_prompt},
                {"role": "user", "content": assessment_text}
            ]
        )

        reply = completion.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        print("ASSISTANT DIRECT ERROR:", e)
        return jsonify({"reply": f"ERROR → {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
