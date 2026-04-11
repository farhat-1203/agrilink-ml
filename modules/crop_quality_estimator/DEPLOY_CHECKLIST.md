# Render Deployment Checklist

## ✅ Pre-Deployment Checklist

### Files Ready
- [ ] `main.py` - FastAPI application entry point
- [ ] `render.yaml` - Render configuration
- [ ] `Procfile` - Backup start command
- [ ] `requirements-render.txt` - CPU-optimized dependencies
- [ ] `.gitignore` - Excludes unnecessary files
- [ ] `models/best_model.pt` - Model checkpoint (5.4 MB)
- [ ] `modules/inference.py` - Inference engine
- [ ] `config.py` - Configuration file
- [ ] All module files in `modules/` and `api/`

### GitHub Setup
- [ ] GitHub account created
- [ ] New repository created (public for free tier)
- [ ] Git initialized locally
- [ ] All files committed
- [ ] Pushed to GitHub main branch

### Render Account
- [ ] Render account created at https://render.com
- [ ] GitHub connected to Render

## 🚀 Deployment Steps

### 1. Push to GitHub
```powershell
cd modules\crop_quality_estimator
.\deploy.ps1
```

Or manually:
```powershell
git init
git add .
git commit -m "Initial deployment"
git remote add origin https://github.com/YOUR_USERNAME/crop-quality-api.git
git push -u origin main
```

### 2. Create Web Service on Render
- [ ] Go to https://render.com/dashboard
- [ ] Click "New +" → "Web Service"
- [ ] Connect your repository
- [ ] Verify auto-detected settings from render.yaml
- [ ] Click "Create Web Service"

### 3. Wait for Build
- [ ] Watch build logs (5-10 minutes)
- [ ] Check for errors
- [ ] Wait for "Live" status

### 4. Test Deployment
- [ ] Visit `https://your-app.onrender.com/health`
- [ ] Check API docs at `https://your-app.onrender.com/docs`
- [ ] Test image upload endpoint

## 🧪 Testing Commands

### Health Check
```bash
curl https://your-app.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": 15,
  "inference_mode": "nn+heuristic"
}
```

### Test Image Analysis
```bash
curl -X POST https://your-app.onrender.com/analyze-image \
  -F "file=@test_image.jpg" \
  -F "crop_hint=tomato"
```

### List Classes
```bash
curl https://your-app.onrender.com/classes
```

## 🔧 Post-Deployment

### Optional: Keep Service Alive
- [ ] Sign up at https://uptimerobot.com
- [ ] Create HTTP monitor
- [ ] URL: `https://your-app.onrender.com/health`
- [ ] Interval: 5 minutes

### Monitor Performance
- [ ] Check Render dashboard for metrics
- [ ] Monitor response times
- [ ] Check memory usage (should be < 512 MB)

### Share with Team
- [ ] Document your API URL
- [ ] Share API documentation link
- [ ] Provide example requests

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Cold start | 15-20 seconds |
| Warm request | 50-100 ms |
| Memory usage | ~400 MB |
| Model size | 5.4 MB |
| Build time | 5-10 minutes |

## ⚠️ Common Issues

### Build Fails
- Check `requirements-render.txt` is being used
- Verify PyTorch CPU index URL is included
- Check build logs for specific errors

### Out of Memory
- Free tier has 512 MB RAM
- Model + dependencies use ~400 MB
- Should work fine, but if issues occur, upgrade to Starter plan

### Slow First Request
- Normal after spin-down (free tier)
- Model loads on first request (~15 seconds)
- Use UptimeRobot to prevent spin-down

### Health Check Timeout
- Render expects response within 30 seconds
- Model loading takes ~10-15 seconds
- Should work, but check logs if issues occur

## 🎯 Success Criteria

- [ ] Service shows "Live" status in Render
- [ ] `/health` endpoint returns 200 OK
- [ ] `/docs` shows Swagger UI
- [ ] Image upload works correctly
- [ ] Response includes quality grade, shelf life, disease detection
- [ ] Response time < 2 seconds for warm requests

## 📞 Support Resources

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- FastAPI Docs: https://fastapi.tiangolo.com
- GitHub Issues: Create issues in your repo

## 🎉 You're Done!

Once all checkboxes are complete, your Crop Quality API is live and ready to use!

**Your API URL:** `https://your-app-name.onrender.com`

**API Documentation:** `https://your-app-name.onrender.com/docs`

Share this with your team and start analyzing crop quality! 🌾
