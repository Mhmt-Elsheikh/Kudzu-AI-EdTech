from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)
client = Client("ysharma/Chat_with_Meta_llama3_8b")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    
    # Check if 'message' key is in the request JSON
    if 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    message = data['message']
    
    # Example parameters for client.predict
    result = client.predict(
        message=message,
        request=0.95,
        param_3=512,
        api_name="/chat"
    )
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,port=4000)
