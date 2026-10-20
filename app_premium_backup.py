
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "enron_spam_data.csv"

HOST = "127.0.0.1"
PORT = 5000

df = None
vectorizer = None
model = None
metrics = {}
confusion = [[0, 0], [0, 0]]


def train_model():
    global df, vectorizer, model, metrics, confusion

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("\n" + "=" * 52)
    print("       SENTINEL AI - INITIALIZING MODEL")
    print("=" * 52)

    df = pd.read_csv(DATA_PATH)

    required = {"Subject", "Message", "Spam/Ham"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")

    df["Subject"] = df["Subject"].fillna("").astype(str)
    df["Message"] = df["Message"].fillna("").astype(str)
    df["Spam/Ham"] = (
        df["Spam/Ham"].astype(str).str.strip().str.lower()
    )

    df["Email_Text"] = (
        df["Subject"].str.strip() + " " + df["Message"].str.strip()
    ).str.strip()

    df = df[
        df["Email_Text"].str.len().gt(0)
        & df["Spam/Ham"].isin(["ham", "spam"])
    ].copy()

    df["Label"] = df["Spam/Ham"].map({"ham": 0, "spam": 1}).astype(int)

    if df["Label"].nunique() != 2:
        raise ValueError("Dataset must contain both ham and spam labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["Email_Text"],
        df["Label"],
        test_size=0.20,
        random_state=42,
        stratify=df["Label"],
    )

    print(f"Usable emails: {len(df):,}")
    print("Converting emails into TF-IDF features...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)

    print("Training Multinomial Naive Bayes...")
    model = MultinomialNB()
    model.fit(train_vectors, y_train)

    predictions = model.predict(test_vectors)

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
        "precision": round(
            precision_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "recall": round(
            recall_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "f1": round(f1_score(y_test, predictions, zero_division=0) * 100, 2),
        "train_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }

    confusion = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist()

    print(f"Accuracy: {metrics['accuracy']}%")
    print(f"Precision: {metrics['precision']}%")
    print(f"Recall: {metrics['recall']}%")
    print(f"F1-score: {metrics['f1']}%")
    print("Model ready.\n")


PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#050a14">
<title>Sentinel AI — Security Intelligence</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root {
  color-scheme: dark;
  --bg:#050a14;
  --panel:#0a1423;
  --panel2:#0d1c30;
  --border:rgba(105,163,211,.19);
  --muted:#8ca3bf;
  --text:#e8f2ff;
  --cyan:#4ce9de;
  --blue:#519bff;
  --green:#32e6a0;
  --red:#ff5878;
  --purple:#b9a1ff;
}
* {box-sizing:border-box}
html {scroll-behavior:smooth;scroll-padding-top:100px}
body {
  margin:0;
  background:
    radial-gradient(ellipse at 75% -15%,rgba(22,88,150,.25),transparent 43%),
    radial-gradient(ellipse at 0% 55%,rgba(15,130,124,.07),transparent 40%),
    var(--bg);
  color:var(--text);
  font-family:Inter,"Segoe UI",Arial,sans-serif;
  font-size:14px;
}
button,textarea {font:inherit}
button {cursor:pointer}
button:focus-visible,a:focus-visible,textarea:focus-visible {
  outline:2px solid var(--cyan);outline-offset:3px
}
::selection {background:#1a858c;color:white}
.topbar {
  height:78px;padding:0 30px;display:flex;align-items:center;
  justify-content:space-between;gap:16px;
  position:sticky;top:0;z-index:20;
  border-bottom:1px solid var(--border);
  background:rgba(5,10,20,.88);backdrop-filter:blur(20px)
}
.brand {display:flex;align-items:center;gap:13px;min-width:0}
.logo {
  width:45px;height:45px;display:grid;place-items:center;
  color:var(--cyan);font-size:25px;border-radius:15px;
  border:1px solid #267b87;
  background:linear-gradient(145deg,#103a4b,#0a1728);
  box-shadow:0 0 25px #35dccc20,inset 0 0 15px #35dccc12;
  animation:logoGlow 4s ease-in-out infinite
}
.brand h1 {margin:0;font-size:18px;letter-spacing:.1px}
.brand p {margin:5px 0 0;color:var(--muted);font-size:11px}
.server-status {
  display:flex;align-items:center;gap:9px;color:var(--green);
  font-size:12px;white-space:nowrap
}
.status-dot {
  height:8px;width:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 12px var(--green);animation:pulse 2s infinite
}
.layout {display:grid;grid-template-columns:214px minmax(0,1fr);min-height:calc(100vh - 78px)}
.sidebar {
  position:sticky;top:78px;height:calc(100vh - 78px);
  padding:23px 13px;border-right:1px solid var(--border);
  display:flex;flex-direction:column;background:#060d18a8
}
.nav-label {font-size:10px;color:#607c9b;letter-spacing:1.7px;padding:0 12px 12px}
.nav {
  display:flex;align-items:center;gap:12px;padding:12px 13px;
  margin-bottom:6px;border:1px solid transparent;border-radius:10px;
  text-decoration:none;color:#9eb2cc;font-size:12px;
  transition:background .25s,color .25s,border .25s,transform .25s
}
.nav:hover {background:#10243a;color:white;transform:translateX(2px)}
.nav.active {
  color:white;background:linear-gradient(105deg,#1260b3,#102c52);
  border-color:#2b74bb66;box-shadow:inset 3px 0 var(--cyan)
}
.nav-icon {font-size:17px;width:22px;text-align:center;color:var(--cyan)}
.sidebar-bottom {
  margin-top:auto;border:1px solid var(--border);border-radius:13px;
  padding:15px 12px;background:linear-gradient(145deg,#0b2030,#08111e);
  text-align:center;color:var(--muted);font-size:11px;line-height:1.9
}
.shield {font-size:27px;color:var(--green);filter:drop-shadow(0 0 10px #23e6a044)}
.sidebar-bottom strong {display:block;color:var(--green);font-size:12px}
main {min-width:0;padding:25px 26px 18px}
.welcome {
  position:relative;overflow:hidden;display:flex;align-items:center;
  justify-content:space-between;gap:18px;padding:23px 25px;
  margin-bottom:17px;border-radius:16px;
  border:1px solid #28608480;
  background:linear-gradient(115deg,#102945,#0b1829 68%,#0a1b2c);
}
.welcome:after {
  content:"";position:absolute;width:230px;height:230px;border-radius:50%;
  right:-80px;top:-120px;border:1px solid #4ce9de18;
  box-shadow:0 0 0 28px #4ce9de05,0 0 0 56px #4ce9de04;
  pointer-events:none
}
.welcome h2 {font-size:20px;margin:0 0 8px}
.welcome p {color:#a6bdd7;font-size:12px;line-height:1.7;margin:0;max-width:650px}
.pill {
  position:relative;z-index:1;border:1px solid #26866a80;
  border-radius:30px;padding:9px 12px;color:var(--green);
  background:#08291f88;font-size:10px;letter-spacing:.5px;white-space:nowrap
}
.cards {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin-bottom:16px}
.card,.panel {
  border:1px solid var(--border);border-radius:14px;min-width:0;
  background:linear-gradient(145deg,rgba(13,29,48,.96),rgba(7,16,29,.98));
  box-shadow:0 12px 28px #0000000b;
  transition:transform .25s,border-color .25s,box-shadow .25s
}
.card {padding:17px 18px;animation:enter .55s both}
.card:nth-child(2){animation-delay:.07s}
.card:nth-child(3){animation-delay:.14s}
.card:nth-child(4){animation-delay:.21s}
.card:hover,.panel:hover {border-color:#3e799b66;box-shadow:0 12px 30px #0000001c}
.card:hover {transform:translateY(-3px)}
.card-top {display:flex;justify-content:space-between;align-items:center;gap:8px}
.card-label {color:var(--muted);font-size:10px;letter-spacing:.75px}
.card-icon {
  width:32px;height:32px;display:grid;place-items:center;
  border-radius:10px;background:#12324a;color:var(--cyan);font-size:17px
}
.card-value {font-size:clamp(23px,2.1vw,30px);font-weight:760;margin:12px 0 7px;letter-spacing:-.7px}
.card-foot {font-size:10px;color:var(--muted)}
.cyan {color:var(--cyan)} .green {color:var(--green)}
.red {color:var(--red)} .purple {color:var(--purple)}
.grid2 {display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-bottom:15px}
.panel {padding:19px;scroll-margin-top:100px}
.panel-head {display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:18px}
.panel h3 {margin:0;font-size:13px;font-weight:650}
.panel-sub {color:var(--muted);font-size:10px;white-space:nowrap}
.chart-wrap {height:238px;position:relative}
.prediction-grid {display:grid;grid-template-columns:1.15fr .85fr;gap:15px}
textarea {
  display:block;width:100%;min-height:145px;resize:vertical;
  padding:14px;color:var(--text);background:#050d18;
  border:1px solid #23405b;border-radius:11px;line-height:1.65;
  outline:none;transition:border .2s,box-shadow .2s
}
textarea:focus {border-color:var(--cyan);box-shadow:0 0 0 3px #4ce9de12}
textarea::placeholder {color:#627e9d}
.input-meta {display:flex;justify-content:space-between;gap:10px;margin-top:7px;color:#66809d;font-size:10px}
.actions {display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}
.btn {
  border:1px solid transparent;border-radius:9px;padding:11px 15px;
  font-size:12px;font-weight:650;transition:transform .2s,filter .2s,box-shadow .2s
}
.btn:hover {transform:translateY(-2px);filter:brightness(1.1)}
.btn:disabled {opacity:.6;cursor:wait;transform:none}
.btn-primary {color:#04141d;background:linear-gradient(100deg,#4ce9de,#4599ff);box-shadow:0 5px 20px #369fe01a}
.btn-secondary {color:#c3d6ee;background:#12243a;border-color:#29445f}
.result {
  min-height:116px;margin-top:15px;padding:16px;
  border:1px solid var(--border);border-radius:11px;background:#081522;
  transition:background .3s,border-color .3s
}
.result.ham {border-color:#1d7658;background:linear-gradient(120deg,#082a21,#081923)}
.result.spam {border-color:#87354b;background:linear-gradient(120deg,#2b101c,#15111e)}
.result-title {color:var(--muted);font-size:10px;letter-spacing:1px;margin-bottom:9px}
.result-main {font-size:19px;font-weight:750}
.result-detail {font-size:11px;color:#9ab0ca;margin-top:9px;line-height:1.6;overflow-wrap:anywhere}
.scan-line {
  height:2px;width:100%;background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  box-shadow:0 0 14px var(--cyan);animation:scan 1.2s linear infinite
}
.spinner {
  display:inline-block;width:13px;height:13px;border:2px solid #24475d;
  border-top-color:var(--cyan);border-radius:50%;
  animation:spin .7s linear infinite;vertical-align:middle;margin-right:7px
}
.matrix-legend {display:flex;gap:15px;flex-wrap:wrap;color:var(--muted);font-size:10px;margin:0 0 10px}
.legend-dot {display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}
.table-panel {margin-top:15px}
.table-wrap {overflow-x:auto}
table {width:100%;border-collapse:collapse;font-size:11px}
th,td {padding:12px 10px;text-align:left;border-bottom:1px solid #182b40}
th {font-weight:550;letter-spacing:.4px;font-size:9px;color:#8199b6;white-space:nowrap}
td {color:#cbd9eb}
td.preview {max-width:350px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tag {display:inline-block;border-radius:6px;padding:5px 8px;font-size:9px;font-weight:750}
.tag.ham {color:var(--green);background:#0b382b}
.tag.spam {color:var(--red);background:#401a28}
.error {color:#ff9caf;font-size:11px;margin-top:10px;overflow-wrap:anywhere}
.footer {text-align:center;padding:21px 0 4px;color:#617b99;font-size:10px}
.footer span {color:var(--cyan)}
@keyframes pulse {0%,100%{opacity:1;box-shadow:0 0 12px var(--green)}50%{opacity:.5;box-shadow:0 0 3px var(--green)}}
@keyframes logoGlow {0%,100%{box-shadow:0 0 18px #35dccc15}50%{box-shadow:0 0 28px #35dccc32}}
@keyframes enter {from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes scan {from{transform:translateY(0);opacity:.3}50%{opacity:1}to{transform:translateY(0);opacity:.3}}
@keyframes spin {to{transform:rotate(360deg)}}
@media(max-width:1100px) {
  .layout {grid-template-columns:175px minmax(0,1fr)}
  main {padding:19px}.sidebar {padding:18px 9px}
  .cards {grid-template-columns:repeat(2,minmax(0,1fr))}
  .card-value {font-size:27px}
}
@media(max-width:800px) {
  .layout {grid-template-columns:1fr}
  .sidebar {height:auto;position:sticky;top:78px;z-index:10;flex-direction:row;
    overflow-x:auto;padding:8px;border-right:0;border-bottom:1px solid var(--border);
    background:#07101ff2}
  .nav-label,.sidebar-bottom {display:none}
  .nav {flex:0 0 auto;margin:0 3px;padding:10px}
  .nav:hover {transform:none}
  .grid2,.prediction-grid {grid-template-columns:1fr}
  .chart-wrap {height:220px}
}
@media(max-width:520px) {
  .topbar {height:68px;padding:0 13px}
  .brand h1 {font-size:14px}.brand p {font-size:9px}
  .logo {width:37px;height:37px}
  .server-status {font-size:0}.status-dot {width:8px;height:8px}
  .layout {min-height:calc(100vh - 68px)}
  .sidebar {top:68px}
  main {padding:12px}
  .welcome {align-items:flex-start;flex-direction:column;padding:18px}
  .welcome h2 {font-size:18px}
  .cards {gap:9px}
  .card {padding:13px}
  .card-value {font-size:22px}
  .card-label {font-size:9px}
  .panel {padding:14px}
  .panel-head {align-items:flex-start}
  .panel-sub {white-space:normal;text-align:right}
  .chart-wrap {height:210px}
  .btn {padding:11px 12px}
}
@media(prefers-reduced-motion:reduce) {
  *,*::before,*::after {
    animation-duration:.01ms!important;animation-iteration-count:1!important;
    scroll-behavior:auto!important;transition-duration:.01ms!important
  }
}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <div class="logo">⬡</div>
    <div>
      <h1>Sentinel AI <span class="cyan">/</span> Spam Detection</h1>
      <p>Machine Learning Security Console · TF-IDF + Naive Bayes</p>
    </div>
  </div>
  <div class="server-status"><span class="status-dot"></span> Local engine online</div>
</header>

<div class="layout">
  <aside class="sidebar">
    <div class="nav-label">WORKSPACE</div>
    <a class="nav active" href="#overview"><span class="nav-icon">◈</span> Overview</a>
    <a class="nav" href="#distribution"><span class="nav-icon">▤</span> Dataset analytics</a>
    <a class="nav" href="#scanner"><span class="nav-icon">✉</span> Email scanner</a>
    <a class="nav" href="#performance"><span class="nav-icon">⌘</span> Model performance</a>
    <a class="nav" href="#records"><span class="nav-icon">▦</span> Dataset records</a>
    <div class="sidebar-bottom">
      <div class="shield">⛨</div>
      <strong>ML ENGINE READY</strong>
      Model loaded in memory<br>
      Local development mode
    </div>
  </aside>

  <main>
    <section class="welcome" id="overview">
      <div>
        <h2>AI Threat Intelligence</h2>
        <p>Explore your email dataset, review evaluation metrics, and scan text with your trained machine-learning classifier.</p>
      </div>
      <div class="pill">● CLASSIFIER ACTIVE</div>
    </section>

    <section class="cards">
      <article class="card">
        <div class="card-top"><div class="card-label">TOTAL EMAILS</div><div class="card-icon">▤</div></div>
        <div class="card-value cyan" id="total">—</div>
        <div class="card-foot">Usable dataset records</div>
      </article>
      <article class="card">
        <div class="card-top"><div class="card-label">LEGITIMATE</div><div class="card-icon green">✓</div></div>
        <div class="card-value green" id="ham">—</div>
        <div class="card-foot" id="hamPct">Ham / not spam</div>
      </article>
      <article class="card">
        <div class="card-top"><div class="card-label">SPAM RECORDS</div><div class="card-icon red">⚠</div></div>
        <div class="card-value red" id="spam">—</div>
        <div class="card-foot" id="spamPct">Spam label records</div>
      </article>
      <article class="card">
        <div class="card-top"><div class="card-label">TEST ACCURACY</div><div class="card-icon purple">⌁</div></div>
        <div class="card-value purple" id="accuracy">—</div>
        <div class="card-foot">Held-out test set</div>
      </article>
    </section>

    <section class="grid2">
      <article class="panel" id="distribution">
        <div class="panel-head"><h3>Dataset distribution</h3><span class="panel-sub">HAM / SPAM</span></div>
        <div class="chart-wrap"><canvas id="distributionChart"></canvas></div>
      </article>
      <article class="panel" id="performance">
        <div class="panel-head"><h3>Model performance</h3><span class="panel-sub">TEST METRICS</span></div>
        <div class="chart-wrap"><canvas id="metricsChart"></canvas></div>
      </article>
    </section>

    <section class="prediction-grid">
      <article class="panel" id="scanner">
        <div class="panel-head"><h3>Live email scanner</h3><span class="panel-sub">REAL MODEL OUTPUT</span></div>
        <textarea id="emailText" maxlength="30000"
          aria-label="Email subject and message"
          placeholder="Paste an email subject and message here..."></textarea>
        <div class="input-meta"><span>Text is processed by this local app</span><span id="charCount">0 / 30000</span></div>
        <div class="actions">
          <button class="btn btn-primary" id="predictBtn" onclick="predictEmail()">⟶ Analyze email</button>
          <button class="btn btn-secondary" onclick="clearScanner()">Clear input</button>
        </div>
        <div id="result" class="result" aria-live="polite">
          <div class="result-title">ANALYSIS RESULT</div>
          <div class="result-main">Awaiting email input</div>
          <div class="result-detail">Enter email text and select Analyze email.</div>
        </div>
        <div id="error" class="error" role="alert"></div>
      </article>

      <article class="panel">
        <div class="panel-head"><h3>Confusion matrix</h3><span class="panel-sub">TEST PREDICTIONS</span></div>
        <div class="matrix-legend">
          <span><i class="legend-dot" style="background:#4ce9de"></i>Predicted ham</span>
          <span><i class="legend-dot" style="background:#ff5878"></i>Predicted spam</span>
        </div>
        <div class="chart-wrap"><canvas id="matrixChart"></canvas></div>
        <div id="metricsText" class="result-detail">Loading model metrics...</div>
      </article>
    </section>

    <section class="panel table-panel" id="records">
      <div class="panel-head"><h3>Dataset records</h3><span class="panel-sub">SAMPLE OF STORED DATA · NOT LIVE TRAFFIC</span></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>ID</th><th>SUBJECT</th><th>MESSAGE PREVIEW</th><th>LABEL</th></tr></thead>
          <tbody id="recordsBody"><tr><td colspan="4">Loading dataset records...</td></tr></tbody>
        </table>
      </div>
    </section>
    <footer class="footer">SENTINEL AI <span>•</span> Educational ML project <span>•</span> Local development server</footer>
  </main>
</div>

<script>
let charts = [];
const fmt = n => Number(n).toLocaleString();
const baseOptions = {
  responsive:true,
  maintainAspectRatio:false,
  animation:{duration:850,easing:'easeOutQuart'},
  plugins:{
    legend:{labels:{color:'#b5c8df',boxWidth:10,boxHeight:10,padding:15,font:{size:10}}}
  }
};
const axisColor = '#91a8c4';
const gridColor = 'rgba(113,151,190,.13)';

async function loadDashboard() {
  const errorBox = document.getElementById('error');
  try {
    const response = await fetch('/api/stats');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Statistics unavailable.');

    document.getElementById('total').textContent = fmt(data.total);
    document.getElementById('ham').textContent = fmt(data.ham);
    document.getElementById('spam').textContent = fmt(data.spam);
    document.getElementById('accuracy').textContent = data.metrics.accuracy + '%';
    document.getElementById('hamPct').textContent = data.ham_pct + '% of dataset';
    document.getElementById('spamPct').textContent = data.spam_pct + '% of dataset';

    charts.push(new Chart(document.getElementById('distributionChart'), {
      type:'doughnut',
      data:{labels:['Ham / legitimate','Spam'],datasets:[{
        data:[data.ham,data.spam],
        backgroundColor:['#32e6a0','#ff5878'],
        borderColor:'#0a1423',borderWidth:5,hoverOffset:10
      }]},
      options:{...baseOptions,cutout:'72%'}
    }));

    charts.push(new Chart(document.getElementById('metricsChart'), {
      type:'bar',
      data:{labels:['Accuracy','Precision','Recall','F1-score'],datasets:[{
        label:'Score (%)',
        data:[data.metrics.accuracy,data.metrics.precision,data.metrics.recall,data.metrics.f1],
        backgroundColor:['#519bff','#4ce9de','#32e6a0','#b9a1ff'],
        borderRadius:6,borderSkipped:false
      }]},
      options:{...baseOptions,scales:{
        x:{ticks:{color:axisColor,font:{size:10}},grid:{display:false}},
        y:{beginAtZero:true,max:100,ticks:{color:axisColor,callback:v=>v+'%'},
          grid:{color:gridColor}}
      }}
    }));

    charts.push(new Chart(document.getElementById('matrixChart'), {
      type:'bar',
      data:{labels:['Actual ham','Actual spam'],datasets:[
        {label:'Predicted ham',data:[data.confusion[0][0],data.confusion[1][0]],backgroundColor:'#4ce9de',borderRadius:5},
        {label:'Predicted spam',data:[data.confusion[0][1],data.confusion[1][1]],backgroundColor:'#ff5878',borderRadius:5}
      ]},
      options:{...baseOptions,indexAxis:'y',scales:{
        x:{beginAtZero:true,ticks:{color:axisColor},grid:{color:gridColor}},
        y:{ticks:{color:axisColor},grid:{display:false}}
      }}
    }));

    document.getElementById('metricsText').textContent =
      `Training: ${fmt(data.metrics.train_count)} emails. Testing: ${fmt(data.metrics.test_count)} emails. ` +
      `TF-IDF vocabulary: ${fmt(data.metrics.vocabulary_size)} terms.`;

    const body = document.getElementById('recordsBody');
    body.replaceChildren();

    data.records.forEach(row => {
      const tr = document.createElement('tr');
      [row.id,row.subject,row.preview].forEach((value,index) => {
        const td = document.createElement('td');
        td.textContent = value || (index === 1 ? '(no subject)' : '');
        if (index === 2) td.className = 'preview';
        tr.appendChild(td);
      });
      const labelTd = document.createElement('td');
      const tag = document.createElement('span');
      tag.className = 'tag ' + (row.label === 'spam' ? 'spam' : 'ham');
      tag.textContent = row.label.toUpperCase();
      labelTd.appendChild(tag);
      tr.appendChild(labelTd);
      body.appendChild(tr);
    });
  } catch (error) {
    errorBox.textContent = 'Dashboard error: ' + error.message;
  }
}

document.getElementById('emailText').addEventListener('input', event => {
  document.getElementById('charCount').textContent =
    event.target.value.length.toLocaleString() + ' / 30000';
});

async function predictEmail() {
  const input = document.getElementById('emailText');
  const result = document.getElementById('result');
  const errorBox = document.getElementById('error');
  const button = document.getElementById('predictBtn');
  const text = input.value.trim();

  errorBox.textContent = '';
  if (!text) {
    errorBox.textContent = 'Enter an email subject or message first.';
    input.focus();
    return;
  }

  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>Scanning...';
  result.className = 'result';
  result.replaceChildren();

  const scanningTitle = document.createElement('div');
  scanningTitle.className = 'result-title';
  scanningTitle.textContent = 'ANALYSIS IN PROGRESS';
  const scanningMain = document.createElement('div');
  scanningMain.className = 'result-main';
  scanningMain.innerHTML = '<span class="spinner"></span>Analyzing email text';
  const scanLine = document.createElement('div');
  scanLine.className = 'scan-line';
  scanLine.style.marginTop = '17px';
  result.append(scanningTitle, scanningMain, scanLine);

  try {
    const response = await fetch('/api/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction failed.');

    result.className = 'result ' + (data.label === 'spam' ? 'spam' : 'ham');
    result.replaceChildren();

    const title = document.createElement('div');
    title.className = 'result-title';
    title.textContent = 'MODEL PREDICTION';

    const heading = document.createElement('div');
    heading.className = 'result-main ' + (data.label === 'spam' ? 'red' : 'green');
    heading.textContent = data.label === 'spam' ? '⚠ SPAM DETECTED' : '✓ HAM / NOT SPAM';

    const detail = document.createElement('div');
    detail.className = 'result-detail';
    detail.textContent =
      `Spam probability estimate: ${data.spam_probability}%. ` +
      `Ham probability estimate: ${data.ham_probability}%. ` +
      'These are model estimates, not a guarantee that an email is safe.';

    result.append(title,heading,detail);
  } catch (error) {
    result.className = 'result';
    result.textContent = 'Prediction error: ' + error.message;
  } finally {
    button.disabled = false;
    button.textContent = '⟶ Analyze email';
  }
}

function clearScanner() {
  document.getElementById('emailText').value = '';
  document.getElementById('charCount').textContent = '0 / 30000';
  document.getElementById('error').textContent = '';
  const result = document.getElementById('result');
  result.className = 'result';
  result.innerHTML =
    '<div class="result-title">ANALYSIS RESULT</div>' +
    '<div class="result-main">Awaiting email input</div>' +
    '<div class="result-detail">Enter email text and select Analyze email.</div>';
}

document.querySelectorAll('.nav').forEach(link => {
  link.addEventListener('click', () => {
    document.querySelectorAll('.nav').forEach(item => item.classList.remove('active'));
    link.classList.add('active');
  });
});

loadDashboard();
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return PAGE


@app.get("/api/stats")
def get_stats():
    ham_count = int((df["Label"] == 0).sum())
    spam_count = int((df["Label"] == 1).sum())
    total = len(df)

    records = []
    for index, row in df.head(8).iterrows():
        records.append({
            "id": str(index),
            "subject": str(row["Subject"]).strip()[:100],
            "preview": str(row["Message"]).strip()[:180],
            "label": str(row["Spam/Ham"]),
        })

    return jsonify({
        "total": total,
        "ham": ham_count,
        "spam": spam_count,
        "ham_pct": round(ham_count / total * 100, 2) if total else 0,
        "spam_pct": round(spam_count / total * 100, 2) if total else 0,
        "metrics": metrics,
        "confusion": confusion,
        "records": records,
    })


@app.post("/api/predict")
def predict_email():
    payload = request.get_json(silent=True) or {}
    email_text = payload.get("text", "")

    if not isinstance(email_text, str):
        return jsonify({"error": "Email text must be a string."}), 400

    email_text = email_text.strip()
    if not email_text:
        return jsonify({"error": "Enter an email subject or message."}), 400

    if len(email_text) > 30000:
        return jsonify({"error": "Maximum email length is 30,000 characters."}), 400

    features = vectorizer.transform([email_text])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    probability_map = {
        int(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    return jsonify({
        "label": "spam" if prediction == 1 else "ham",
        "spam_probability": round(probability_map.get(1, 0) * 100, 2),
        "ham_probability": round(probability_map.get(0, 0) * 100, 2),
    })


if __name__ == "__main__":
    train_model()
    print(f"\nDashboard: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the local server.\n")
    app.run(host=HOST, port=PORT, debug=False)
