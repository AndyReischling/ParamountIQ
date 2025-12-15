# Sentinel Backend - Soccer Video Analysis

A Flask-based backend service that analyzes soccer videos using Google's Gemini AI and computer vision to detect events, track ball/player movement, and provide tactical insights.

## Features

- 🎥 **Video Analysis**: Upload soccer videos for automated event detection
- 🧠 **AI-Powered Insights**: Uses Google Gemini AI for tactical analysis
- 📊 **Statistics Tracking**: Tracks ball and player movement throughout the video
- 🎧 **Audio Detection**: Detects whistles and crowd reactions
- 👀 **Visual Detection**: Identifies scene cuts, replays, and key moments
- 💾 **Caching**: Caches analysis results for faster subsequent requests

## Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## Setup Instructions

### 1. Clone and Navigate to Directory

```bash
cd sentinel-backend
```

### 2. Create Virtual Environment (if not already created)

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

Create a `.env` file in the `sentinel-backend` directory:

```bash
cp .env.example .env
```

Then edit `.env` and add your Google Gemini API key:

```
API_KEY=your_actual_api_key_here
```

**To get an API key:**
1. Visit [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key or use an existing one
4. Copy the key and paste it in your `.env` file

### 6. Run the Server

```bash
python server.py
```

The server will start on `http://localhost:5001`

## Usage

### API Endpoint

**POST** `/analyze`

Upload a video file for analysis:

```bash
curl -X POST -F "video=@path/to/your/video.mp4" http://localhost:5001/analyze
```

### Response Format

```json
{
  "success": true,
  "events": [
    {
      "time": "01:23",
      "title": "Tactical Formation Shift",
      "desc": "Team transitions from 4-3-3 to 4-4-2 defensive block. This creates numerical superiority in midfield.",
      "type": "NORMAL",
      "stats": {
        "possession": 55,
        "possession_team": "Home",
        "passes": 15,
        "passes_team": "Away"
      }
    }
  ],
  "count": 1,
  "video": "video.mp4",
  "cached": false
}
```

### Frontend Integration

See `API_USAGE.md` for detailed examples using JavaScript, React, Python, and cURL.

## Project Structure

```
sentinel-backend/
├── server.py              # Main Flask server
├── pass_a_processor.py    # Audio/visual event detection
├── stats_processor.py     # Ball/player tracking and statistics
├── requirements.txt       # Python dependencies
├── .env                   # API key configuration (create this)
├── .env.example          # Example environment file
├── uploads/               # Uploaded video files (auto-created)
├── cache/                # Cached analysis results (auto-created)
└── stats_cache/          # Statistics cache (auto-created)
```

## API Endpoints

- `GET /` - Serves index.html
- `GET /test` - Serves test.html
- `GET /api` - API information
- `POST /analyze` - Upload and analyze video

## How It Works

1. **Pass A - Local Vision**: 
   - Analyzes audio for whistles and crowd reactions
   - Detects visual scene cuts and replays
   - Creates a manifest of timestamps of interest

2. **Pass B - Gemini AI**:
   - Uploads video to Google Gemini
   - Analyzes specific timestamps identified in Pass A
   - Generates tactical insights and event descriptions

3. **Background Processing**:
   - Tracks ball and player movement
   - Computes statistics (speed, distance, possession)
   - Updates cache with statistics

## Troubleshooting

### API Key Issues

If you see warnings about the API key:
- Make sure `.env` file exists in `sentinel-backend/` directory
- Verify the API key starts with `AIza`
- Check that the key is valid at [https://ai.dev/usage](https://ai.dev/usage)

### Quota Exceeded

The server automatically tries multiple Gemini models:
- Starts with free-tier models (`gemini-1.5-flash`)
- Falls back to other models if quota is exceeded
- Check your usage at [https://ai.dev/usage?tab=rate-limit](https://ai.dev/usage?tab=rate-limit)

### Video Processing Errors

- Ensure video format is supported (MP4, MOV, AVI, etc.)
- Large videos may take several minutes to process
- Check server logs for detailed error messages

## Development

The server runs in debug mode by default. To run in production:

```python
# In server.py, change:
app.run(debug=True, port=5001)
# To:
app.run(debug=False, host='0.0.0.0', port=5001)
```

## License

[Add your license here]


