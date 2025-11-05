from flask import Flask, request, jsonify, render_template_string
import boto3, joblib, os, io

app = Flask(__name__)

# --- Load model from S3 ---

BUCKET = os.getenv("MODEL_BUCKET", "<placeholder>")
KEY    = os.getenv("MODEL_KEY", "<placeholder>.pkl")
REGION = os.getenv("AWS_REGION", "<placeholder>")
bucket = os.getenv("MODEL_BUCKET", "<placeholder>")
key = os.getenv("MODEL_KEY", "<placeholder>.pkl")

s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "<placeholder>"))
obj = s3.get_object(Bucket=bucket, Key=key)
model = joblib.load(io.BytesIO(obj["Body"].read()))

# --- Routes ---
@app.route("/")
def home():
    html = """
    <!doctype html><meta charset="utf-8">
    <title>Movie Sentiment</title>
    <style>
    body{font-family:Arial,Helvetica,sans-serif;max-width:760px;margin:40px auto;padding:0 16px}
    textarea{width:100%;height:140px}
    button{padding:10px 16px;margin-top:10px}
    .result{margin-top:12px;font-weight:600}
    </style>
    <h1>Movie Review Sentiment</h1>
    <textarea id="t" placeholder="Type your review..."></textarea><br>
    <button onclick="go()">Analyze</button>
    <div id="o" class="result"></div>
    <script>
    async function go(){
      const out=document.getElementById('o');
      const text=document.getElementById('t').value.trim();
      if(!text){out.textContent='Please enter text.';return;}
      out.textContent='Analyzing...';
      try{
        const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
        const j=await r.json();
        if(!r.ok) throw new Error(j.error||('HTTP '+r.status));
        out.textContent=`Sentiment: ${j.sentiment}` + (j.confidence?` (conf ${(j.confidence*100).toFixed(1)}%)`:'');
      }catch(e){ out.textContent='Error: '+e.message; }
    }
    </script>
    """
    return render_template_string(html)
_bundle = None
def get_bundle():
    global _bundle
    if _bundle is None:
        s3 = boto3.client("s3", region_name=REGION)
        obj = s3.get_object(Bucket=BUCKET, Key=KEY)
        _bundle = joblib.load(io.BytesIO(obj["Body"].read()))
    return _bundle


@app.route("/health")
def health():
    return "ok"
@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify(error="Missing 'text'"), 400

        bundle = get_bundle()
        tfidf = bundle["tfidf"]
        clf   = bundle["clf"]
        X = tfidf.transform([text])
        pred = clf.predict(X)[0]
        conf = float(clf.predict_proba(X)[0].max()) if hasattr(clf, "predict_proba") else None

        return jsonify(sentiment=str(pred), confidence=conf)
    except Exception as e:
        app.logger.exception("predict failed")
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
