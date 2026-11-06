"""DermaLens AI - stylish Streamlit dashboard.   Run:  streamlit run app.py"""
import json, os
import numpy as np, pandas as pd, torch, streamlit as st
import plotly.express as px, plotly.graph_objects as go
from PIL import Image
from model import load_checkpoint, eval_transform, CLASS_NAMES
from explain import grad_cam, attention_rollout, overlay

st.set_page_config(page_title="DermaLens AI", page_icon="🔬", layout="wide")

PALETTE = ["#7c4dff", "#00e5ff", "#ff4081", "#ffd740", "#69f0ae", "#ff6e40", "#40c4ff"]
MALIGNANT = {"mel", "bcc", "akiec"}          # malignant / pre-malignant groups in HAM10000

# --------------------------------------------------------------------------- CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 600px at 8% -10%, rgba(124,77,255,.28), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(0,229,255,.16), transparent 60%),
    radial-gradient(800px 500px at 50% 110%, rgba(255,64,129,.14), transparent 60%),
    linear-gradient(180deg, #0b1020, #0e1330 60%, #0b1020);
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; max-width: 1320px; }

/* sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(190deg, #1a1450 0%, #0d1128 55%, #0a0f22 100%);
  border-right: 1px solid rgba(255,255,255,.08);
}
.brand { font-weight: 800; font-size: 1.5rem; letter-spacing: .5px;
  background: linear-gradient(90deg,#00e5ff,#7c4dff,#ff4081); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }
.brand-sub { color: #9aa6d9; font-size: .8rem; margin-bottom: 1rem; }

/* hero */
.hero {
  position: relative; overflow: hidden; padding: 2.4rem 2.6rem; border-radius: 26px; margin-bottom: 1.4rem;
  background: linear-gradient(120deg, rgba(106,17,203,.55), rgba(37,117,252,.40), rgba(0,229,255,.28));
  border: 1px solid rgba(255,255,255,.18); box-shadow: 0 20px 60px rgba(60,40,200,.35);
  backdrop-filter: blur(14px);
}
.hero:before { content:""; position:absolute; width:420px; height:420px; right:-120px; top:-160px;
  background: radial-gradient(circle, rgba(255,64,129,.55), transparent 65%); filter: blur(10px); }
.hero h1 { margin: 0; font-size: 3.4rem; font-weight: 800; line-height: 1.05;
  background: linear-gradient(90deg,#ffffff,#9fe8ff 40%,#ffb3d1); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }
.hero .tag { margin: .5rem 0 1rem; color: #dbe3ff; font-size: 1.1rem; font-weight: 500; }
.chip { display:inline-block; margin: 0 8px 8px 0; padding: 6px 14px; border-radius: 999px; font-size: .78rem;
  font-weight: 600; color: #e8ecff; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,.06); border-radius: 14px; padding: 10px 20px;
  border: 1px solid rgba(255,255,255,.08); color: #b8c1ec; font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(90deg,#7c4dff,#00b0ff) !important; color: #fff !important;
  box-shadow: 0 8px 24px rgba(124,77,255,.45); border-color: transparent; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* cards */
.card { background: linear-gradient(145deg, rgba(255,255,255,.09), rgba(255,255,255,.03));
  border: 1px solid rgba(255,255,255,.12); border-radius: 20px; padding: 1.2rem 1.4rem;
  box-shadow: 0 10px 30px rgba(0,0,0,.25); backdrop-filter: blur(10px); }
.kpi { text-align: center; padding: 1.2rem .8rem; border-radius: 20px; position: relative; overflow: hidden;
  border: 1px solid rgba(255,255,255,.14); box-shadow: 0 12px 32px rgba(0,0,0,.3); }
.kpi .v { font-size: 2.2rem; font-weight: 800; color: #fff; text-shadow: 0 2px 14px rgba(0,0,0,.35); }
.kpi .l { font-size: .78rem; text-transform: uppercase; letter-spacing: 1.5px; color: rgba(255,255,255,.85); font-weight: 600; }
.g1 { background: linear-gradient(135deg,#7c4dff,#3f51b5); }
.g2 { background: linear-gradient(135deg,#00b0ff,#00e5ff 130%); }
.g3 { background: linear-gradient(135deg,#ff4081,#ff6e40); }
.g4 { background: linear-gradient(135deg,#00c853,#00e5a0); }

/* result */
.result-name { font-size: 1.9rem; font-weight: 800; line-height: 1.15; margin: .2rem 0 .6rem;
  background: linear-gradient(90deg,#fff,#9fe8ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.badge { display:inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: .8rem; margin-right: 8px; }
.b-mal { background: rgba(255,77,109,.18); color: #ff8fa3; border: 1px solid rgba(255,77,109,.55); }
.b-ben { background: rgba(0,230,118,.16); color: #69f0ae; border: 1px solid rgba(0,230,118,.5); }
.b-info { background: rgba(124,77,255,.2); color: #c7b6ff; border: 1px solid rgba(124,77,255,.5); }
.top3 { display:flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed rgba(255,255,255,.12); color:#d6dcff; }
.top3 b { color: #00e5ff; }

/* pipeline */
.pipe { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin: 8px 0 18px; }
.step { padding: 12px 16px; border-radius: 14px; font-weight: 600; color:#fff; font-size:.9rem;
  background: linear-gradient(135deg,#7c4dff,#00b0ff); box-shadow: 0 8px 20px rgba(124,77,255,.35); }
.arrow { color:#00e5ff; font-size:1.3rem; font-weight:800; }

.section-title { font-weight: 800; font-size: 1.25rem; margin: .6rem 0 .6rem; color: #fff; }
.small { color:#9aa6d9; font-size:.82rem; }
[data-testid="stFileUploader"] section { border: 2px dashed rgba(124,77,255,.6); border-radius: 18px;
  background: rgba(124,77,255,.07); }
div.stButton > button { background: linear-gradient(90deg,#7c4dff,#00b0ff); color:#fff; border:0; border-radius:12px; font-weight:600; }
[data-testid="stImage"] img { border-radius: 16px; box-shadow: 0 10px 28px rgba(0,0,0,.4); }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- helpers
def style(fig, h=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Poppins", color="#d6dcff"), margin=dict(l=10, r=10, t=55, b=10),
                      height=h, colorway=PALETTE, title_font=dict(size=16, color="#ffffff"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    return fig


def kpi(label, value, grad):
    return f'<div class="kpi {grad}"><div class="v">{value}</div><div class="l">{label}</div></div>'


def title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


dev = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def get_model(path):
    return load_checkpoint(path, dev)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="brand">🔬 DermaLens AI</div><div class="brand-sub">Explainable skin-lesion intelligence</div>',
                unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    out_dir = st.text_input("Outputs folder", "outputs")
    ckpt = st.text_input("Checkpoint", os.path.join(out_dir, "best_model.pt"))
    data_dir = st.text_input("Dataset folder (for EDA)", "data/ham10000")
    alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.5)
    st.markdown("---")
    st.markdown('<div class="small">⚠️ Research prototype only.<br>Not a medical device – not for diagnosis.</div>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- hero
st.markdown("""
<div class="hero">
  <h1>DermaLens AI</h1>
  <div class="tag">See what the model sees — explainable skin-lesion classification with a hybrid CNN-Transformer.</div>
  <span class="chip">🧠 EfficientNet-B0 + Transformer</span>
  <span class="chip">🔥 Grad-CAM</span>
  <span class="chip">👁️ Attention Rollout</span>
  <span class="chip">🧬 HAM10000 · 7 classes</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🩺  Predict & Explain", "📊  Model Performance", "🗂  Dataset Explorer", "ℹ️  About"])

# --------------------------------------------------------------------------- tab 1
with tab1:
    if not os.path.exists(ckpt):
        st.warning("Checkpoint not found. Train first: `python train.py ...` then reload this page.")
    else:
        model, classes, cfg = get_model(ckpt)
        up = st.file_uploader("Drop a dermoscopic image here", type=["jpg", "jpeg", "png"])
        if not up:
            st.markdown('<div class="card small">⬆️ Upload an image to get the prediction, confidence, '
                        'Grad-CAM and Transformer attention maps.</div>', unsafe_allow_html=True)
        else:
            img = Image.open(up).convert("RGB")
            x = eval_transform(cfg["img_size"])(img).unsqueeze(0).to(dev)
            _, probs, top = grad_cam(model, x)
            choice = st.selectbox("Explain which class?", ["Predicted class"] + [f"{c} – {CLASS_NAMES[c]}" for c in classes])
            idx = top if choice == "Predicted class" else classes.index(choice.split(" – ")[0])
            cam, _, _ = grad_cam(model, x, idx)
            rollout, _ = attention_rollout(model, x)
            base = np.asarray(img.resize((cfg["img_size"],) * 2)) / 255.0
            code, conf = classes[top], float(probs[top])

            left, right = st.columns([1.15, 1])
            with left:
                title("🖼️ Visual explanation")
                c1, c2, c3 = st.columns(3)
                c1.image(base, caption="Input", use_container_width=True)
                c2.image(overlay(base, cam, alpha), caption="Grad-CAM · CNN", use_container_width=True)
                c3.image(overlay(base, rollout, alpha), caption="Attention · Transformer", use_container_width=True)
                st.markdown('<div class="small">Warm colours (red/yellow) = regions the model relied on most.</div>',
                            unsafe_allow_html=True)
            with right:
                risk = ('<span class="badge b-mal">⚠ Malignant / pre-malignant group</span>' if code in MALIGNANT
                        else '<span class="badge b-ben">✔ Benign group</span>')
                order = np.argsort(probs)[::-1][:3]
                rows = "".join(f'<div class="top3"><span>{classes[i]} – {CLASS_NAMES[classes[i]]}</span>'
                               f'<b>{probs[i]*100:.1f}%</b></div>' for i in order)
                st.markdown(f'<div class="card"><div class="small">MODEL PREDICTION</div>'
                            f'<div class="result-name">{CLASS_NAMES[code]}</div>{risk}'
                            f'<span class="badge b-info">{code}</span><div style="margin-top:14px">{rows}</div></div>',
                            unsafe_allow_html=True)
                g = go.Figure(go.Indicator(mode="gauge+number", value=conf * 100, number={"suffix": "%", "font": {"size": 38}},
                    title={"text": "Confidence", "font": {"size": 14}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#00e5ff"}, "bgcolor": "rgba(255,255,255,.06)",
                           "steps": [{"range": [0, 50], "color": "rgba(255,64,129,.35)"},
                                     {"range": [50, 75], "color": "rgba(255,215,64,.30)"},
                                     {"range": [75, 100], "color": "rgba(105,240,174,.30)"}]}))
                st.plotly_chart(style(g, 230), use_container_width=True)

            title("📈 Class probabilities")
            dfp = pd.DataFrame({"class": [f"{c} – {CLASS_NAMES[c]}" for c in classes], "probability": probs}).sort_values("probability")
            fig = px.bar(dfp, x="probability", y="class", orientation="h", color="probability",
                         color_continuous_scale=["#2b2f77", "#7c4dff", "#00e5ff"], range_x=[0, 1])
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(style(fig, 360), use_container_width=True)
            if conf < 0.6:
                st.info("Low confidence – treat this prediction with extra caution.")

# --------------------------------------------------------------------------- tab 2
with tab2:
    mp = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(mp):
        st.info("metrics.json not found – run train.py first.")
    else:
        m = json.load(open(mp))
        auc = f"{m['macro_auc_ovr']:.3f}" if m.get("macro_auc_ovr") else "n/a"
        k = st.columns(4)
        k[0].markdown(kpi("Accuracy", f"{m['accuracy']*100:.1f}%", "g1"), unsafe_allow_html=True)
        k[1].markdown(kpi("Balanced acc.", f"{m['balanced_accuracy']*100:.1f}%", "g2"), unsafe_allow_html=True)
        k[2].markdown(kpi("Macro F1", f"{m['macro_f1']:.3f}", "g3"), unsafe_allow_html=True)
        k[3].markdown(kpi("Macro AUC", auc, "g4"), unsafe_allow_html=True)
        st.write("")
        rep = pd.DataFrame(m["report"]).T.loc[m["classes"], ["precision", "recall", "f1-score", "support"]]
        a, b = st.columns(2)
        cm = np.array(m["confusion_matrix"])
        fig = px.imshow(cm, x=m["classes"], y=m["classes"], text_auto=True, aspect="auto",
                        color_continuous_scale=["#0b1020", "#3d2a8f", "#7c4dff", "#00e5ff"],
                        labels=dict(x="Predicted", y="True"), title="Confusion matrix (test set)")
        a.plotly_chart(style(fig, 430), use_container_width=True)
        f1 = rep.reset_index().rename(columns={"index": "class"})
        fig = px.bar(f1.melt(id_vars="class", value_vars=["precision", "recall", "f1-score"]), x="class", y="value",
                     color="variable", barmode="group", title="Per-class precision / recall / F1",
                     color_discrete_sequence=["#7c4dff", "#00e5ff", "#ff4081"])
        b.plotly_chart(style(fig, 430), use_container_width=True)
        if "mel" in rep.index:
            st.markdown(f'<div class="card"><b>🎯 Melanoma recall:</b> {rep.loc["mel","recall"]*100:.1f}% '
                        f'<span class="small">— the most clinically important number: missed melanomas are the costliest error.</span></div>',
                        unsafe_allow_html=True)
        h = pd.DataFrame(m["history"])
        c, d = st.columns(2)
        c.plotly_chart(style(px.line(h, x="epoch", y=["val_acc", "val_macro_f1"], title="Validation curves", markers=True), 340), use_container_width=True)
        d.plotly_chart(style(px.area(h, x="epoch", y="train_loss", title="Training loss"), 340), use_container_width=True)
        title("📋 Detailed report")
        st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"})
                     .background_gradient(cmap="viridis", subset=["precision", "recall", "f1-score"]), use_container_width=True)

# --------------------------------------------------------------------------- tab 3
with tab3:
    mp = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(mp):
        st.info("Point the sidebar to the folder containing HAM10000_metadata.csv.")
    else:
        df = pd.read_csv(mp)
        df["class"] = df.dx.map(CLASS_NAMES)
        k = st.columns(4)
        k[0].markdown(kpi("Images", f"{len(df):,}", "g1"), unsafe_allow_html=True)
        k[1].markdown(kpi("Unique lesions", f"{df.lesion_id.nunique():,}", "g2"), unsafe_allow_html=True)
        k[2].markdown(kpi("Classes", f"{df.dx.nunique()}", "g3"), unsafe_allow_html=True)
        k[3].markdown(kpi("Mean age", f"{df.age.mean():.0f} yrs", "g4"), unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        vc = df["class"].value_counts().reset_index()
        a.plotly_chart(style(px.bar(vc, x="count", y="class", orientation="h", color="class", title="Class distribution (imbalanced!)")
                             .update_layout(showlegend=False), 400), use_container_width=True)
        b.plotly_chart(style(px.pie(df, names="dx", hole=.55, title="Class share"), 400), use_container_width=True)
        c, d = st.columns(2)
        c.plotly_chart(style(px.histogram(df, x="age", color="dx", nbins=20, title="Age by class"), 380), use_container_width=True)
        d.plotly_chart(style(px.histogram(df, x="localization", color="dx", title="Body site by class")
                             .update_xaxes(tickangle=-35), 380), use_container_width=True)
        e, f = st.columns(2)
        e.plotly_chart(style(px.pie(df, names="sex", hole=.55, title="Sex"), 320), use_container_width=True)
        f.plotly_chart(style(px.pie(df, names="dx_type", hole=.55, title="Diagnosis confirmation type"), 320), use_container_width=True)

# --------------------------------------------------------------------------- tab 4
with tab4:
    title("🏗️ Architecture")
    st.markdown("""
<div class="pipe">
 <div class="step">Dermoscopic image</div><div class="arrow">➜</div>
 <div class="step">EfficientNet-B0 (local features)</div><div class="arrow">➜</div>
 <div class="step">1×1 projection → tokens + CLS</div><div class="arrow">➜</div>
 <div class="step">4 × Transformer blocks (8 heads)</div><div class="arrow">➜</div>
 <div class="step">CLS → 7-class head</div>
</div>
""", unsafe_allow_html=True)
    a, b = st.columns(2)
    a.markdown('<div class="card"><b>🔥 Grad-CAM (CNN branch)</b><br><span class="small">Back-propagates the class score to the CNN '
               'feature map to highlight the image regions that most increased that class.</span></div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><b>👁️ Attention Rollout (Transformer)</b><br><span class="small">Multiplies attention matrices across '
               'all Transformer blocks (with residuals) to show where the CLS token looked.</span></div>', unsafe_allow_html=True)
    st.write("")
    c, d = st.columns(2)
    c.markdown('<div class="card"><b>🧬 Data</b><br><span class="small">HAM10000 · 7 classes · split by <code>lesion_id</code> to '
               'prevent leakage · weighted sampling + label smoothing for class imbalance.</span></div>', unsafe_allow_html=True)
    d.markdown('<div class="card"><b>⚠️ Disclaimer</b><br><span class="small">Educational research prototype. Predictions are '
               'not medical advice – always consult a dermatologist.</span></div>', unsafe_allow_html=True)
