# Deploy Crop Quality API to Render

## Prerequisites
- GitHub account
- Render account (free tier works fine)
- Git installed locally

## Step 1: Prepare Your Repository

### 1.1 Create a new GitHub repository
1. Go to https://github.com/new
2. Name it: `crop-quality-api`
3. Make it Public (required for Render free tier)
4. Don't initialize with README (we have our own files)
5. Click "Create repository"

### 1.2 Push your code to GitHub

Open PowerShell in the `crop_quality_estimator` directory:

```powershell
cd C:\Users\Farhat Momin\Desktop\agri-ml\modules\crop_quality_estimator

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Crop Quality API"

# Add your GitHub repo as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/crop-quality-api.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Render

### 2.1 Connect to Render
1. Go to https://render.com
2. Sign up or log in (you can use GitHub to sign in)
3. Click "New +" button in top right
4. Select "Web Service"

### 2.2 Connect Your Repository
1. Click "Connect a repository"
2. If first time: Authorize Render to access your GitHub
3. Find and select your `crop-quality-api` repository
4. Click "Connect"

### 2.3 Configure the Service

Render will auto-detect the `render.yaml` file, but verify these settings:

**Basic Settings:**
- Name: `crop-quality-api` (or your preferred name)
- Region: `Oregon (US West)` (or closest to you)
- Branch: `main`
- Root Directory: Leave blank (or `.` if required)

**Build & Deploy:**
- Runtime: `Python 3`
- Build Command: 
  ```
  pip install --upgrade pip && pip install -r requirements-render.txt
  ```
- Start Command:
  ```
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```

**Plan:**
- Select `Free` (512 MB RAM, spins down after 15 min inactivity)

**Environment Variables:**
- `DEVICE` = `cpu`
- `PYTHON_VERSION` = `3.10.0`

### 2.4 Deploy
1. Click "Create Web Service"
2. Wait 5-10 minutes for the build to complete
3. Watch the logs for any errors

## Step 3: Test Your Deployment

Once deployed, you'll get a URL like: `https://crop-quality-api.onrender.com`

### Test the health endpoint:
```bash
curl https://crop-quality-api.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": 15,
  "inference_mode": "nn+heuristic",
  "device": "cpu"
}
```

### Test image analysis:
```bash
curl -X POST https://crop-quality-api.onrender.com/analyze-image \
  -F "file=@tomato.jpg;type=image/jpeg" \
  -F "crop_hint=tomato"
```

### View API docs:
Open in browser: `https://crop-quality-api.onrender.com/docs`

## Step 4: Keep Your Service Alive (Optional)

Render free tier spins down after 15 minutes of inactivity. To keep it alive:

### Option 1: UptimeRobot (Recommended)
1. Sign up at https://uptimerobot.com (free)
2. Add New Monitor:
   - Monitor Type: HTTP(s)
   - Friendly Name: Crop Quality API
   - URL: `https://crop-quality-api.onrender.com/health`
   - Monitoring Interval: 5 minutes
3. Save

### Option 2: Cron Job
Set up a cron job or scheduled task to ping your endpoint every 14 minutes.

## Troubleshooting

### Build fails with "torch" error
- Make sure you're using `requirements-render.txt` which has CPU-only torch
- Check that the build command includes the PyTorch CPU index URL

### Service crashes with "Out of Memory"
- The model + dependencies use ~400MB
- Free tier has 512MB RAM
- If it crashes, try:
  1. Remove unused dependencies from requirements
  2. Upgrade to Starter plan ($7/month, 512MB → 2GB RAM)

### First request is very slow (20-30 seconds)
- This is normal after spin-down
- The model loads on first request
- Subsequent requests are fast (~50-100ms)
- Solution: Use UptimeRobot to keep it alive

### Health check fails
- Render expects `/health` to respond within 30 seconds
- Our startup loads the model, which takes ~10-15 seconds
- This should work fine, but if it fails:
  1. Check logs for errors
  2. Increase health check timeout in Render dashboard

### "Module not found" errors
- Make sure all files are committed to git
- Check that `modules/`, `api/`, `models/` folders are included
- Verify `.gitignore` isn't excluding necessary files

## Environment Variables

You can add these in Render dashboard under "Environment":

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cpu` | Device for inference (cpu/cuda) |
| `MODEL_PATH` | `models/best_model.pt` | Path to model checkpoint |
| `PORT` | Auto | Port to run on (Render sets this) |

## Updating Your Deployment

To deploy updates:

```bash
# Make your changes
git add .
git commit -m "Update: description of changes"
git push

# Render will automatically rebuild and redeploy
```

## Monitoring

### View Logs
1. Go to Render dashboard
2. Click on your service
3. Click "Logs" tab
4. Watch real-time logs

### Metrics
- Render dashboard shows:
  - CPU usage
  - Memory usage
  - Request count
  - Response times

## Cost Optimization

### Free Tier Limits
- 750 hours/month (enough for 1 service running 24/7)
- Spins down after 15 min inactivity
- 512 MB RAM
- 0.1 CPU

### When to Upgrade
Consider Starter plan ($7/month) if:
- You need 24/7 uptime without spin-down
- You need more RAM (2GB)
- You need faster CPU (0.5 CPU)
- You need custom domain

## Security Best Practices

1. **Don't commit secrets**: Use Render environment variables
2. **Enable CORS properly**: Update `allow_origins` in production
3. **Rate limiting**: Add rate limiting middleware for production
4. **HTTPS**: Render provides free SSL (automatic)

## Support

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- FastAPI Docs: https://fastapi.tiangolo.com

## Next Steps

1. ✅ Deploy to Render
2. ✅ Test all endpoints
3. ✅ Set up UptimeRobot
4. 📱 Integrate with your mobile app
5. 📊 Monitor usage and performance
6. 🔄 Set up CI/CD for automatic deployments

---

**Your API is now live! 🎉**

Share your API URL with your team and start analyzing crop quality!
