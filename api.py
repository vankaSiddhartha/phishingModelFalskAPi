from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from urllib.parse import urlparse
import re
import numpy as np
from collections import Counter

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Load models and resources
model = joblib.load("model/phishing_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

# Load English dictionary words
english_words = set(nltk.corpus.words.words())

# Define safe TLDs and suspicious patterns
SAFE_TLDS = {'.com', '.org', '.net', '.edu', '.gov', '.ai', '.in', '.io',
             '.co', '.us', '.uk', '.ca', '.au', '.de', '.fr', '.jp'}

SUSPICIOUS_PATTERNS = [
    r'ww\d+\.',  # Matches patterns like ww17., ww1., etc.
    r'[0-9]{4,}',  # Four or more consecutive numbers
    r'(jp-video|video-jp)',  # Specific suspicious terms
    r'-stream',
    r'video[0-9]+'
]


def extract_domain_features(domain):
    """Extract features from domain name"""
    features = {
        'length': len(domain),
        'num_digits': sum(c.isdigit() for c in domain),
        'num_hyphens': domain.count('-'),
        'num_dots': domain.count('.'),
        'has_https': 1 if domain.startswith('https://') else 0,
        'has_http': 1 if domain.startswith('http://') else 0,
        'has_suspicious_pattern': any(re.search(pattern, domain.lower()) for pattern in SUSPICIOUS_PATTERNS),
        'is_safe_tld': any(domain.lower().endswith(tld) for tld in SAFE_TLDS)
    }
    return features


def analyze_url_nlp(url):
    """Enhanced NLP analysis of URL"""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc if parsed_url.netloc else parsed_url.path
    path = parsed_url.path

    # Basic tokenization
    domain_tokens = word_tokenize(re.sub(r'[^\w\s]', ' ', domain))
    path_tokens = word_tokenize(re.sub(r'[^\w\s]', ' ', path))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    domain_tokens = [word.lower()
                     for word in domain_tokens if word.lower() not in stop_words]
    path_tokens = [word.lower()
                   for word in path_tokens if word.lower() not in stop_words]

    # Analyze domain characteristics
    domain_features = extract_domain_features(domain)
    brand_similarities = analyze_brand_similarity(domain)
    entropy = calculate_entropy(domain)
    homograph_attacks = detect_homograph_attack(domain)

    analysis = {
        "risk_factors": [],
        "domain_analysis": {
            "length": domain_features['length'],
            "entropy": entropy,
            "features": domain_features
        }
    }

    # Enhanced risk assessment
    if not domain_features['is_safe_tld']:
        analysis["risk_factors"].append(
            {"type": "unsafe_tld", "severity": "critical"})

    if domain_features['has_suspicious_pattern']:
        analysis["risk_factors"].append(
            {"type": "suspicious_pattern", "severity": "critical"})

    if domain_features['num_digits'] > 2:
        analysis["risk_factors"].append(
            {"type": "excessive_numbers", "severity": "high"})

    # Original risk assessments remain...
    if domain_features['length'] > 30:
        analysis["risk_factors"].append({"type": "length", "severity": "high"})

    if len(brand_similarities) > 0:
        analysis["risk_factors"].append(
            {"type": "brand_impersonation", "severity": "critical"})

    if len(homograph_attacks) > 0:
        analysis["risk_factors"].append(
            {"type": "homograph_attack", "severity": "critical"})

    return analysis


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'url' not in data:
            return jsonify({"error": "No URL provided"}), 400

        url = data["url"]

        # Always perform NLP analysis first for suspicious TLDs and patterns
        nlp_analysis = analyze_url_nlp(url)

        # Check if there are any critical severity risk factors
        has_critical_risks = any(factor["severity"] == "critical"
                                 for factor in nlp_analysis["risk_factors"])

        # If critical risks found, mark as phishing immediately
        if has_critical_risks:
            return jsonify({
                "success": True,
                "url": url,
                "is_phishing": True,
                "confidence_score": 1.0,
                "nlp_analysis": nlp_analysis
            })

        # Otherwise, proceed with ML prediction
        features = vectorizer.transform([url])
        ml_prediction = model.predict(features)[0]

        # Calculate final confidence score
        risk_severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }

        if nlp_analysis["risk_factors"]:
            confidence_score = sum(risk_severity_weights[factor["severity"]]
                                   for factor in nlp_analysis["risk_factors"]) / len(nlp_analysis["risk_factors"])
        else:
            confidence_score = 0.0

        return jsonify({
            "success": True,
            "url": url,
            "is_phishing": ml_prediction or confidence_score > 0.5,
            "confidence_score": min(confidence_score, 1.0),
            "nlp_analysis": nlp_analysis
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
