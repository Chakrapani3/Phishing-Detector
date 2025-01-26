import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import load
from flask import Flask, request, jsonify, render_template
import pickle

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the LSTM CNN model for phishing URL detection
phishing_url_model = load_model('phishing_url_model.h5')

# Load the SVM model and TF-IDF vectorizer for email content detection
svm_model = load('svm_model.pkl')
tfidf_vectorizer = load('tfidf_vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect-phishing', methods=['POST'])
def detect_phishing():
    try:
        data = request.json
        email_content = data.get('email_content', '')
        url_content = data.get('url_content', '')

        if not email_content:
            return jsonify({"error": "No email content provided"}), 400

        # Detect phishing email content
        email_vector = tfidf_vectorizer.transform([email_content])
        email_prediction = svm_model.predict(email_vector)[0]

        result = {
            "email_prediction": int(email_prediction)
        }

        if email_prediction == 1:
            result['message'] = "This email is phishing. Please move it to spam."
            return jsonify(result)

        # Ask for URL if email is legitimate
        if url_content:
            # Detect phishing URL content
            url_vector = tokenizer.texts_to_sequences([url_content])
            url_vector = tf.keras.preprocessing.sequence.pad_sequences(url_vector, maxlen=200)
            url_prediction = phishing_url_model.predict(url_vector)[0][0]
            result['url_prediction'] = int(url_prediction > 0.5)

            if result['url_prediction']:
                result['message'] = "The URL is phishing. Please delete the email."
            else:
                result['message'] = "The email is legitimate."
        else:
            result['message'] = "The email is legitimate. Please provide a URL to check."

        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
