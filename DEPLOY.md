# 🚀 Deployment Guide — Sammy Quant Lab

## Step 1 · Push to GitHub

```bash
cd sammy-quant-lab

git init
git add .
git commit -m "feat: initial release — Sammy Quant Lab v2.0"

# Create repo on github.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/sammy-quant-lab.git
git branch -M main
git push -u origin main
```

---

## Step 2 · Deploy Dashboard → Streamlit Cloud (Free)

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/sammy-quant-lab`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
5. Click **"Deploy!"**
6. Your live URL will be: `https://YOUR_USERNAME-sammy-quant-lab-dashboard-app-XXXXX.streamlit.app`

Update the `index.html` and `README.md` with this URL.

---

## Step 3 · Enable GitHub Pages (Portfolio Landing Page)

1. In your repo → **Settings** → **Pages**
2. Under **Source**, select **"GitHub Actions"**
3. The `deploy-pages.yml` workflow will run on every push to `main`
4. Your portfolio page will be at: `https://YOUR_USERNAME.github.io/sammy-quant-lab/`

---

## Step 4 · Update Branding

Replace `YOUR_USERNAME` in:
- `README.md` (3 places)
- `index.html` (4 places — GitHub link, LinkedIn link, demo link)

```bash
# Quick find-and-replace (Linux/Mac):
sed -i 's/YOUR_USERNAME/your-actual-github-username/g' README.md index.html
sed -i 's/YOUR_PROFILE/your-linkedin-profile/g' index.html
sed -i 's|https://your-app.streamlit.app|https://your-actual-streamlit-url.streamlit.app|g' README.md index.html
```

---

## Step 5 · (Optional) Add to LinkedIn

Add this to your LinkedIn profile:

**Project: Sammy Quant Lab**
> Multi-model FX forecasting engine: ARIMA · GARCH · LSTM · Hybrid models.
> End-to-end data pipeline, interactive Streamlit dashboard, walk-forward backtesting.
> Tech: Python, PyTorch, statsmodels, Plotly, Streamlit, GitHub Actions.

---

## Checklist

- [ ] Repo pushed to GitHub
- [ ] Streamlit Cloud deployed
- [ ] GitHub Pages enabled  
- [ ] `YOUR_USERNAME` replaced everywhere
- [ ] LinkedIn updated
- [ ] README badges working (CI badge from GitHub Actions)
