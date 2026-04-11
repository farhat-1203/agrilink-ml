# Crop Quality Estimator API

AI-powered crop quality analysis API using MobileNetV3 disease detection + OpenCV freshness estimation.

## 🚀 Quick Deploy to Render

**Time: 10 minutes | Cost: FREE**

### Step 1: Push to GitHub
```powershell
cd modules\crop_quality_estimator
.\deploy.ps1
```

### Step 2: Deploy on Render
1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Click "Create Web Service"
5. Wait 5-10 minutes for build

### Step 3: Test
```bash
curl https://your-app.onrender.com/health
```

**Full Guide:** See [RENDER_DEPLOY.md](RENDER_DEPLOY.md)

## 📡 API Endpoints

### POST /analyze-image
```bash
curl -X POST https://your-app.onrender.com/analyze-image \
  -F "file=@tomato.jpg" \
  -F "crop_hint=tomato"
```

Returns:
- Quality grade (A/B/C)
- Shelf life (days)
- Disease detection
- Freshness score (0-100)
- Market recommendation

### GET /docs
Interactive API documentation at `https://your-app.onrender.com/docs`

## 🎯 What It Analyzes

✅ Quality Grade (A/B/C)  
✅ Shelf Life Estimate  
✅ 15+ Disease Types  
✅ Freshness Score (0-100)  
✅ Market Tier Recommendation  

## 📊 Performance

- **Inference**: 50-100ms per image
- **Memory**: ~400 MB
- **Model**: 5.4 MB
- **Accuracy**: 95%+

## 🔧 Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Visit: http://localhost:8000/docs

## 📚 Documentation

- [RENDER_DEPLOY.md](RENDER_DEPLOY.md) - Complete deployment guide
- [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md) - Deployment checklist
- [DEPLOYMENT.md](DEPLOYMENT.md) - General deployment info

## 🆘 Need Help?

- Check [RENDER_DEPLOY.md](RENDER_DEPLOY.md) for troubleshooting
- Create a GitHub issue
- Visit https://render.com/docs

---

**Ready?** Run `.\deploy.ps1` to get started! 🚀
