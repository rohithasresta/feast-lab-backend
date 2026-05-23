# FEAST Lab RAG Backend
**Built by Rohitha Sresta Ganji**

## Deploy to Railway (5 minutes)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "FEAST Lab RAG backend"
gh repo create feast-lab-backend --public --push
```

### Step 2 — Deploy on Railway
1. Go to https://railway.app and sign in with GitHub
2. Click **New Project → Deploy from GitHub repo**
3. Select `feast-lab-backend`
4. Railway auto-detects the Procfile and deploys

### Step 3 — Add your OpenAI API key
1. In Railway dashboard → your project → **Variables**
2. Add: `OPENAI_API_KEY` = `sk-...` (your key)
3. Railway redeploys automatically

### Step 4 — Get your public URL
Railway gives you a URL like `https://feast-lab-backend-production.up.railway.app`

### Step 5 — Update the frontend
In `feast_lab_homepage.html`, find this line:
```js
const API = 'http://localhost:8000';
```
Replace with your Railway URL:
```js
const API = 'https://feast-lab-backend-production.up.railway.app';
```

That's it — the chatbot now works from anywhere, 24/7.

## File structure
```
feast_lab_deploy/
├── main.py           ← FastAPI app
├── requirements.txt  ← Python dependencies
├── Procfile          ← Railway start command
├── README.md
└── docs/             ← Lab documents (auto-loaded on startup)
    ├── research_overview.txt
    ├── publications.txt
    └── participation_faq.txt
```

## Adding new documents
Just add PDF or TXT files to the `docs/` folder, commit, and push.
Railway redeploys and the new docs are automatically embedded.
