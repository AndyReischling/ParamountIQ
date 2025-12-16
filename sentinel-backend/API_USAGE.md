# API Usage Guide

## Endpoint: `/analyze`

### Request
- **Method**: `POST`
- **URL**: `http://localhost:5001/analyze`
- **Content-Type**: `multipart/form-data`
- **Body**: Form data with field name `video` containing the video file

### Response Format

**Success Response (200 OK):**
```json
{
  "success": true,
  "events": [
    {
      "time": "01:23",
      "title": "Touchdown Scored",
      "desc": "WR #12 catches 45-yard pass in end zone",
      "type": "TOUCHDOWN"
    },
    {
      "time": "05:45",
      "title": "Penalty Called",
      "desc": "Defensive holding penalty on cornerback",
      "type": "PENALTY"
    }
  ],
  "count": 2,
  "video": "video.mp4"
}
```

**Error Response (400/401/500):**
```json
{
  "error": "Error message here",
  "message": "Detailed error description"
}
```

### Event Types
- `TOUCHDOWN` - A touchdown was scored
- `PENALTY` - A penalty was called
- `TIMEOUT` - A timeout was called
- `TURNOVER` - A turnover occurred (fumble, interception)
- `NORMAL` - Regular play or replay

## Frontend Integration Examples

### JavaScript (Fetch API)
```javascript
async function analyzeVideo(videoFile) {
  const formData = new FormData();
  formData.append('video', videoFile);

  try {
    const response = await fetch('http://localhost:5001/analyze', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Analysis failed');
    }

    // Handle response
    const events = data.events || data; // Supports both formats
    console.log(`Found ${events.length} events`);
    
    events.forEach(event => {
      console.log(`${event.time} - ${event.type}: ${event.title}`);
    });

    return events;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
```

### React Example
```jsx
import { useState } from 'react';

function VideoAnalyzer() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await fetch('http://localhost:5001/analyze', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      const eventList = data.events || data;
      setEvents(eventList);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept="video/*" onChange={handleVideoUpload} />
      {loading && <p>Analyzing...</p>}
      {events.map((event, index) => (
        <div key={index}>
          <h3>{event.time} - {event.title}</h3>
          <p>{event.desc}</p>
          <span className={`badge ${event.type}`}>{event.type}</span>
        </div>
      ))}
    </div>
  );
}
```

### cURL Example
```bash
curl -X POST \
  http://localhost:5001/analyze \
  -F "video=@path/to/your/video.mp4"
```

### Python Example
```python
import requests

def analyze_video(video_path):
    url = 'http://localhost:5001/analyze'
    
    with open(video_path, 'rb') as video_file:
        files = {'video': video_file}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        events = data.get('events', data)  # Supports both formats
        return events
    else:
        raise Exception(f"Error: {response.json()}")
```

## CORS

The server has CORS enabled, so you can call it from any frontend running on any port/domain.

## Notes

- The analysis can take several minutes depending on video length
- The API processes videos up to the limits of the Gemini API
- Make sure your API key is set in the `.env` file
