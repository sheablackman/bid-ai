# Deployment Guide

## Quick Deploy Options

### Option 1: Render (Recommended - Free & Easy)

1. **Create a Render account** at https://render.com (free signup)

2. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

3. **Deploy on Render**:
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: bid-ai (or any name)
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn backend:app --host 0.0.0.0 --port $PORT`
   - Click "Create Web Service"

4. **Set Environment Variables** in Render dashboard:
   - `OPENAI_API_KEY`: Your OpenAI API key (required)
   - `LLM_PROVIDER`: `openai` (or `ollama` if using local)
   - `OPENAI_MODEL`: `gpt-4o` (or your preferred model)

5. **Get your URL**: 
   - Render will provide: `https://your-app-name.onrender.com`
   - Free tier apps spin down after 15min inactivity (takes ~30s to wake up)

---

### Option 2: Railway (Alternative - Free Tier)

1. **Sign up** at https://railway.app
2. **Click "New Project" → "Deploy from GitHub repo"**
3. **Select your repository**
4. **Set environment variables** (same as Render)
5. **Deploy automatically**
6. **Get URL**: Railway provides a unique URL like `your-app.up.railway.app`

---

### Option 3: Fly.io (Free Tier)

1. **Install Fly CLI**: `curl -L https://fly.io/install.sh | sh`
2. **Sign up**: `fly auth signup`
3. **Initialize**: `fly launch` (in your project directory)
4. **Set secrets**: `fly secrets set OPENAI_API_KEY=your-key`
5. **Deploy**: `fly deploy`
6. **Get URL**: `your-app.fly.dev`

---

## Important Notes

- **OpenAI API Key Required**: You'll need a valid OpenAI API key with credits
- **Free Tier Limitations**: 
  - Render: Spins down after 15min (slow first load)
  - Railway: Limited hours/month
  - Fly.io: Limited resources
- **For Production/Resume**: Consider upgrading to a paid tier ($7-20/month) for better performance

## Testing Your Deployment

Once deployed, test:
- Visit the root URL: Should show the main interface
- Upload files: Test file upload functionality
- Generate contract: Test the full workflow

## Resume Link Format

Use: `https://your-app-name.onrender.com` (or your deployment URL)
