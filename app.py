from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your LLM model (replace with your model loading code)
##model = load_llm_model()

@app.route('/generate_text', methods=['POST'])
def generate_text(name):
    input_text = request.json.get(name)
    # generated_text = model.generate_text(input_text)
    # return jsonify({'generated_text': generated_text})
    return jsonify({'name': input_text})
    print(input_text)

# Add more endpoints as needed

if __name__ == '__main__':
    app.run(debug=True)  # For development