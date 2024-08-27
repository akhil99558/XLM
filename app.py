from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model_path = "model_save"  

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    from_tf=False,  # Assuming PyTorch
    use_safetensors=True
)

# Define the class labels
class_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Define request body
class TextRequest(BaseModel):
    text: str

# Serve the HTML interface
@app.get("/", response_class=HTMLResponse)
def get_interface():
    html_content = """
    <html>
        <head>
            <title>Text Classification</title>
            <style>
                body {
                    background-color: black;
                    color: white;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                }
                h1 {
                    color: yellow;
                }
                form {
                    display: inline-block;
                    margin: 20px;
                }
                input[type="text"] {
                    background-color: yellow;
                    color: black;
                    border: none;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                button {
                    background-color: yellow;
                    color: black;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #ffcc00;
                }
                #result {
                    margin: 20px;
                }
                p {
                    margin: 5px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Text Classification Interface</h1>
                <form action="/predict/" method="post">
                    <label for="text">Enter text:</label>
                    <input type="text" id="text" name="text" required>
                    <button type="submit">Predict</button>
                </form>
                <div id="result"></div>
            </div>
            <script>
                const form = document.querySelector('form');
                form.addEventListener('submit', async (event) => {
                    event.preventDefault();
                    const formData = new FormData(form);
                    const text = formData.get('text');
                    
                    const response = await fetch('/predict/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text }),
                    });
                    
                    const result = await response.json();
                    document.getElementById('result').innerHTML = `
                        <h2>Prediction Result</h2>
                        <p><strong>Label:</strong> ${result.label}</p>
                        <p><strong>Confidence:</strong> ${result.confidence}</p>
                    `;
                });
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Define prediction endpoint
@app.post("/predict/")
async def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_label_idx = predictions.argmax().item()
    confidence = predictions.max().item()

    predicted_label = class_labels[predicted_label_idx]

    return {"label": predicted_label, "confidence": confidence}

# Run the app with uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
