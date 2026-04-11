# 🚀 Deploy Your Crop Quality API to Render

## ✅ Everything is Ready!

All files needed for Render deployment are configured and ready to go.

## 📋 Quick Start (3 Steps)

### Step 1: Push to GitHub (5 minutes)

Open PowerShell in this directory and run:

```powershell
.\deploy.ps1
```

This script will:
- Initialize git
- Ask for your GitHub repository URL
- Commit all files
- Push to GitHub

**Don't have a GitHub repo yet?**
1. Go to https://github.com/new
2. Name it: `crop-quality-api`
3. Make it **Public** (required for Render free tier)
4. Click "Create repository"
5. Copy the URL and paste it when deploy.ps1 asks

### Step 2: Deploy on Render (2 minutes)

1. Go to https://render.com (sign up if needed)
2. Click **"New +"** → **"Web Service"**
3. Click **"Connect a repository"**
4. Select your `crop-quality-api` repository
5. Click **"Connect"**
6. Render will auto-detect `render.yaml` - just click **"Create Web Service"**

### Step 3: Wait & Test (5-10 minutes)

Watch the build logs. Once it shows **"Live"**:

```bash
# Test health endpoint
curl https://your-app.onrender.com/health

# View API docs
# Open in browser: https://your-app.onrender.com/docs
```

## 🎉 That's It!

Your API is now live and ready to analyze crop images!

## 📖 Need More Details?

- **Complete Guide**: [RENDER_DEPLOY.md](RENDER_DEPLOY.md)
- **Checklist**: [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md)
- **Troubleshooting**: See RENDER_DEPLOY.md section

## 🧪 Test Your API

### Using curl:
```bash
curl -X POST https://your-app.onrender.com/analyze-image \
  -F "file=@tomato.jpg" \
  -F "crop_hint=tomato"
```

### Using Python:
```python
import requests

with open("tomato.jpg", "rb") as f:
    files = {"file": f}
    data = {"crop_hint": "tomato"}
    response = requests.post(
        "https://your-app.onrender.com/analyze-image",
        files=files,
        data=data
    )
    print(response.json())
```

## 📊 What You Get

Your API will return:
- ✅ Quality Grade (A/B/C)
- ✅ Shelf Life (days)
- ✅ Disease Detection (15+ types)
- ✅ Freshness Score (0-100)
- ✅ Market Recommendation
- ✅ 6 CV Signals (saturation, greenness, browning, etc.)

## 💰 Cost

**FREE** on Render free tier:
- 512 MB RAM
- 750 hours/month
- Auto-deploy from GitHub
- Free SSL certificate

**Note**: Service spins down after 15 min inactivity. Use UptimeRobot (free) to keep it alive.

## ⚡ Performance

- **Cold start**: 15-20 seconds (after spin-down)
- **Warm request**: 50-100 ms
- **Memory usage**: ~400 MB
- **Model size**: 5.4 MB

## 🆘 Common Issues

### "git: command not found"
Install Git: https://git-scm.com/download/win

### "Authentication failed"
```powershell
git config --global credential.helper wincred
```

### Build fails on Render
- Check that `requirements-render.txt` is being used
- View build logs for specific error
- See troubleshooting in RENDER_DEPLOY.md

### Service crashes
- Free tier has 512 MB RAM (should be enough)
- Check logs in Render dashboard
- Model + dependencies use ~400 MB

## 📞 Support

- **Render Docs**: https://render.com/docs
- **Render Community**: https://community.render.com
- **Create Issue**: On your GitHub repo

## 🎯 Next Steps After Deployment

1. ✅ Test all endpoints
2. ✅ Set up UptimeRobot to prevent spin-down
3. ✅ Share API URL with your team
4. ✅ Integrate with your mobile app
5. ✅ Monitor usage in Render dashboard

---

## 🚀 Ready to Deploy?

Run this command now:

```powershell
.\deploy.ps1
```

Then follow the prompts!

**Questions?** Read [RENDER_DEPLOY.md](RENDER_DEPLOY.md) for the complete guide.

---

**Good luck! Your crop quality API will be live in 10 minutes! 🌾**
