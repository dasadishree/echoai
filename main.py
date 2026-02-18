# api
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import shutil
import os
from recognize_speaker import recognize

app=FastAPI()
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/identify")
async def identify_speaker(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        name, confidence, all_scores = recognize("speaker_profiles.pkl", file_path)

        return {
            "name": name.replace("_", " "),
            "confidence": f"{confidence:.2%}",
            "all_scores": {k: f"{v:.2%}" for k, v in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)}
        }
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <html>
        <head>
            <title>EchoAI</title>
            <style>
                body { 
                    font-family: sans-serif;
                    background-color: #f4f7f6;
                    margin: 0;
                    display: flex;
                    flex-direction: column; 
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                }
                .container{
                    background: white;
                    padding: 40px;
                    border-radius: 15px;
                    box-shadown: 0 10px 25px rgba(0,0,0,0.1);
                    text-align: center;
                    width: 90%;
                    max-width: 800px;
                }
                h1{
                    color: #2c3e50;
                    margin-bottom: 30px;
                }
                input[type="file"] {
                    margin: 20px 0;
                }
                button {
                    background-color: #3498db;
                    color:white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    transition: background 0.3s;
                }
                button:hover {
                    background-color: #2980b9;
                }
                details {
                    margin-top: 20px;
                    background: #f9f9f9;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #ddd;
                }
                summary {
                    font-weight: bold;
                    cursor: pointer;
                    color: #34495e;
                    outline: none;
                }
                .score-grid{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 10px;
                    margin-top: 20px;
                    text-align: left;
                    font-family: monospace;
                    font-size: 14px;
                }
                .score-item {
                    display: flex;
                    justify-content: space-between;
                    padding: 5px 10px;
                    background: white;
                    border-bottom: 1px solid #eee;
                }
                .score-name{ 
                    color: #555;
                }
                .score-val{
                    font-weight: bold;
                    color: #2c3e50;
                }
            </style>
        </head>

        <body>
            <div class="container">
                <h1>EchoAI</h1>
                <input type="file" id="audioInput" accept=".wav">
                <button id="submitBtn" onclick="uploadFile()">Identify</button>

                <div id="result-area" style="display:none">
                    <h2 id="greeting"></h2>
                    <p id="topConf"></p>

                    <details id="infoDropdown">
                        <summary>See Details</summary>
                        <div id="scoreGrid" class="score-grid"></div>
                    </details>
                </div>
            </div>
        
            <script>
                async function uploadFile() {
                    const fileInput = document.getElementById('audioInput');
                    const btn = document.getElementById('submitBtn');

                    if(!fileInput.files[0]) return alert("Please select a file first!");

                    btn.disabled = true;
                    btn.innerText = "Analyzing Audio...";
                    document.getElementById("result-area").style.display = "none";

                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);

                    try{
                        const response = await fetch('/identify', {method: 'POST', body: formData});

                        if(!response.ok) {
                            throw new Error("Server crashed or returned error: "+ response.status);
                        }

                        const data = await response.json();
                    
                        //show result area
                        document.getElementById('greeting').innerText = "Hello " + data.name+"!";
                        document.getElementById("topConf").innerText = "Confidence Score: " + data.confidence;

                        const grid = document.getElementById('scoreGrid');
                        grid.innerHTML = "";

                        Object.entries(data.all_scores).forEach(([name, score]) => {
                            const div = document.createElement("div");
                            div.className = "score-item";
                            div.innerHTML = `<span class="score-name">${name}</span><span class="score-val">${score}</span>`;
                            grid.appendChild(div);
                        });

                        document.getElementById("result-area").style.display = "block";
                    } catch(err){
                        console.error(err);
                        alert("Error processing audio: "+err.message);
                        btn.innerText = "Identify";
                        btn.disabled = false;                        
                    } finally {
                        btn.disabled = false;
                        btn.innerText = "Identify Speaker";
                    }
                }
            </script>
        </body>
    </html>    

    """