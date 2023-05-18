from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    # Process the input text or make predictions using your model
    # Replace this with your actual model inference code
    print(input_text)
    return 'Input Text: {}'.format(input_text)

if __name__ == '__main__':
    app.run(debug=True)
