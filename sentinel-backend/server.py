import os
import json
import time
import re
import hashlib
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from pass_a_processor import NFLEventDetector
from stats_processor import VideoStatisticsProcessor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


app = Flask(__name__)
CORS(app) # CRITICAL: Allows your browser frontend to talk to this python script


# CONFIG
UPLOAD_FOLDER = 'uploads'
CACHE_FOLDER = 'cache'
EDITS_FOLDER = 'cache/edits'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(EDITS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CACHE_FOLDER'] = CACHE_FOLDER
app.config['EDITS_FOLDER'] = EDITS_FOLDER

# Thread lock for cache file operations
cache_lock = threading.Lock()


def get_video_hash(filepath):
    """Generate a hash of the video file for caching"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        # Read first 1MB and file size for quick hashing
        chunk = f.read(1024 * 1024)
        hash_md5.update(chunk)
        f.seek(0, os.SEEK_END)
        hash_md5.update(str(f.tell()).encode())
    return hash_md5.hexdigest()


def get_cache_path(video_hash):
    """Get the cache file path for a video hash"""
    return os.path.join(app.config['CACHE_FOLDER'], f"{video_hash}.json")


def load_cached_analysis(video_hash):
    """Load cached analysis if it exists"""
    cache_path = get_cache_path(video_hash)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                print(f"✅ Loaded cached analysis for video")
                return cached_data
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
    return None


def save_analysis_cache(video_hash, events_data):
    """Save analysis results to cache"""
    cache_path = get_cache_path(video_hash)
    try:
        # Thread-safe file writing
        with cache_lock:
            with open(cache_path, 'w') as f:
                json.dump(events_data, f, indent=2)
        print(f"💾 Saved analysis to cache")
    except Exception as e:
        print(f"⚠️ Error saving cache: {e}")


def update_cache_with_statistics(video_hash, statistics):
    """Update existing cache file with statistics (thread-safe)"""
    cache_path = get_cache_path(video_hash)
    
    try:
        with cache_lock:
            # Load existing cache
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
            else:
                cached_data = {}
            
            # Update with statistics
            cached_data['statistics'] = statistics
            
            # Save updated cache
            with open(cache_path, 'w') as f:
                json.dump(cached_data, f, indent=2)
            
            print(f"✅ Updated cache with statistics")
    except Exception as e:
        print(f"⚠️ Error updating cache with statistics: {e}")


def process_statistics_background(video_path, video_hash):
    """Process video statistics in background thread"""
    def process():
        try:
            print(f"🔄 Starting background statistics processing for {video_hash[:8]}...")
            processor = VideoStatisticsProcessor(video_path)
            # Process every 2nd frame for performance (can be adjusted)
            statistics = processor.process_video(sample_rate=2)
            
            # Update cache with statistics
            update_cache_with_statistics(video_hash, statistics)
            print(f"✅ Background statistics processing complete for {video_hash[:8]}")
        except Exception as e:
            print(f"⚠️ Background statistics processing failed: {e}")
            # Don't raise - this is background processing, shouldn't break the app
    
    # Start background thread
    thread = threading.Thread(target=process, daemon=True)
    thread.start()
    return thread


# --- SETUP API KEY ---
# Load from environment variable (supports .env file or system env)
GEMINI_API_KEY = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ WARNING: API_KEY not found in environment variables.")
    print("💡 Create a .env file in this directory with: API_KEY=your_api_key_here")
    print("   Or set it as an environment variable: export API_KEY=your_api_key_here")
    print("   Get your API key from: https://makersuite.google.com/app/apikey")
else:
    print("✅ API key loaded successfully.")
    # Validate API key format (basic check)
    if not GEMINI_API_KEY.startswith("AIza"):
        print("⚠️ WARNING: API key format looks incorrect. Should start with 'AIza'")

genai.configure(api_key=GEMINI_API_KEY)


def analyze_with_gemini(video_path, manifest):
    print("🚀 Uploading to Gemini 1.5 Pro...")
    
    # 1. Upload Video
    video_file = genai.upload_file(path=video_path)
    
    # 2. Wait for processing
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    print(" Done.")


    if video_file.state.name == "FAILED":
        raise ValueError("Gemini failed to process the video file.")
    
    # Helper function to list available models if needed
    def list_available_models():
        """List available models for debugging"""
        try:
            models = genai.list_models()
            available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            return available
        except Exception as e:
            return f"Error listing models: {str(e)}"


    # 3. Construct Targeted Prompt
    # This is the "Gold Standard" efficiency hack. We only ask about the timestamps we found.
    prompt = "You are a professional NFL analyst providing tactical insights for a premium sports broadcast. "
    prompt += "I have uploaded an NFL football video. Computer vision has flagged specific timestamps of interest. "
    prompt += "For EACH timestamp below, analyze the footage from -2 seconds to +4 seconds around it. "
    prompt += "Focus heavily on TACTICAL FORMATIONS, SCHEMES, and STRATEGIES. Analyze:\n"
    prompt += "- Offensive formations (Shotgun, Pistol, I-Formation, Spread, etc.) and personnel groupings\n"
    prompt += "- Defensive schemes (4-3, 3-4, Nickel, Dime, Cover 2, Cover 3, Man coverage, Zone blitz, etc.)\n"
    prompt += "- Route concepts (Mesh, Levels, Slant-Flat, Post-Corner, etc.) and passing game strategy\n"
    prompt += "- Running game concepts (Inside Zone, Outside Zone, Power, Counter, etc.)\n"
    prompt += "- Blitz packages and pressure schemes (A-gap blitz, Edge rush, Stunt, etc.)\n"
    prompt += "- Pre-snap motion, shifts, and alignment adjustments\n"
    prompt += "- Down and distance strategy and situational football\n"
    prompt += "Provide ANALYTICAL INSIGHTS about WHY these tactical elements matter in this moment.\n\n"
    
    prompt += "TIMESTAMPS TO ANALYZE:\n"
    for event in manifest:
        prompt += f"- {event['seconds']}s (Vision detected: {event['type']})\n"


    prompt += """
    \nRETURN VALID JSON ONLY. No comments, no markdown code blocks. Schema:

    [
      {
        "time": "MM:SS",
        "title": "Short Event Title (max 4 words, tactical focus)",
        "desc": "EXACTLY 2 SENTENCES. First sentence: specific tactical observation (formation, shape, pattern). Second sentence: why it matters strategically. Be punchy and direct.",
        "type": "TOUCHDOWN",
        "stats": {
          "yards": 45,
          "yards_team": "Home",
          "down": 3,
          "distance": 7,
          "completion_pct": 68.5,
          "completion_team": "Away",
          "speed": "22.5 mph",
          "speed_player": "Home WR #12",
          "pressure": "High",
          "pressure_team": "Away"
        }
      }
    ]

    IMPORTANT: 
    - Return ONLY valid JSON array, no markdown, no code blocks, no explanations
    - "time" format: "MM:SS" (e.g., "01:23")
    - "desc" MUST be EXACTLY 2 sentences (punchy, direct, tactical):
      * Sentence 1: Specific tactical observation (formation, scheme, route concept, defensive alignment, etc.)
      * Sentence 2: Strategic implication - why this moment matters tactically (down/distance, field position, game situation)
      * Keep it concise and impactful - no fluff
    - "type" must be one of: "TOUCHDOWN", "PENALTY", "TIMEOUT", "TURNOVER", "NORMAL"
    - "stats" object should include 2-4 of these fields:
      * "yards": number (yards gained/lost on play) - REQUIRES "yards_team": string (team name or "Home"/"Away")
      * "down": number (1-4, current down) - REQUIRES "distance": number (yards to go)
      * "completion_pct": number (0-100, completion percentage) - REQUIRES "completion_team": string (team name or "Home"/"Away")
      * "speed": string (e.g., "22.5 mph" or "10.0 m/s") - REQUIRES "speed_player": string (player description like "Home WR #12" or "Away QB")
      * "pressure": string ("High", "Medium", or "Low") - REQUIRES "pressure_team": string (team applying pressure)
      * "field_position": string (e.g., "Own 25", "Opponent 40", "Red Zone") - REQUIRES "field_position_team": string (team name)
    - All stats MUST specify which team or player they refer to using the corresponding "_team" or "_player" field
    - Team names can be: "Home", "Away", or actual team names if visible (e.g., "Chiefs", "Bills", "49ers")
    - Player descriptions should include team and position/number if visible (e.g., "Home QB #15", "Away WR #88", "Home DE #99")
    - All stats values must be valid JSON (numbers as numbers, strings as strings)
    - Include realistic stats based on what's visible in the video

    """


    print("🧠 Thinking...")
    # Try latest models first, fallback to free-tier if quota issues
    # Latest as of Dec 2024: Gemini 2.5 series (Pro, Flash, Flash-Lite)
    models_to_try = [
        'gemini-2.5-pro',        # Latest: Most capable, best quality (Sep 2025)
        'gemini-2.5-flash',      # Latest: Balanced speed/quality
        'gemini-2.5-flash-lite', # Latest: Cost-optimized variant
        'gemini-1.5-pro',        # Free tier fallback: Better quality
        'gemini-1.5-flash',      # Free tier fallback: Fast, supports video
        'gemini-pro-latest',     # Free tier alternative
        'gemini-flash-latest'    # Free tier alternative
    ]
    
    last_error = None
    for model_name in models_to_try:
        try:
            print(f"🔄 Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
            print(f"✅ Successfully using model: {model_name}")
            break
        except Exception as model_error:
            error_msg = str(model_error)
            last_error = model_error
            
            # Check for quota errors
            if "429" in error_msg or "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                print(f"⚠️ Quota exceeded for {model_name}, trying next model...")
                continue
            
            # Check for model not found errors
            if "not found" in error_msg.lower() or "404" in error_msg:
                print(f"⚠️ Model {model_name} not found, trying next model...")
                continue
            
            # For other errors, try next model
            print(f"⚠️ Error with {model_name}: {error_msg[:100]}... trying next model...")
            continue
    else:
        # All models failed
        if last_error:
            error_msg = str(last_error)
            if "429" in error_msg or "quota" in error_msg.lower():
                available = list_available_models()
                raise ValueError(
                    f"Quota exceeded for all models. Original error: {error_msg}\n"
                    f"Available models: {available}\n"
                    f"Please check your API quota at: https://ai.dev/usage?tab=rate-limit"
                )
            else:
                available = list_available_models()
                raise ValueError(f"All models failed. Last error: {error_msg}\nAvailable models: {available}")
        else:
            raise ValueError("Failed to initialize any model")
    
    # Clean response - handle various formats Gemini might return
    text = response.text.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # Try to extract JSON array if there's extra text
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"📄 Response text (first 500 chars): {text[:500]}")
        # Try to fix common issues
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        # Remove comments (though shouldn't be there)
        text = re.sub(r'//.*', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON response. Error: {e}\nResponse preview: {text[:200]}")


@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')

@app.route('/test', methods=['GET'])
def test():
    return send_from_directory('.', 'test.html')

@app.route('/api/video/<filename>', methods=['GET'])
def get_video_file(filename):
    """Serve video file from uploads folder"""
    try:
        # Security: ensure filename is safe
        safe_filename = secure_filename(filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        if os.path.exists(video_path) and os.path.isfile(video_path):
            return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename)
        else:
            return jsonify({"error": "Video file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({
        "service": "Sentinel Backend - NFL Video Analysis",
        "version": "1.0",
        "endpoints": {
            "/analyze": {
                "method": "POST",
                "description": "Upload a video file for analysis",
                "content_type": "multipart/form-data",
                "field_name": "video",
                "returns": "JSON array of detected events"
            }
        },
        "status": "running"
    })


@app.route('/api/videos', methods=['GET'])
def list_videos():
    """List all cached video analyses"""
    try:
        cache_folder = app.config['CACHE_FOLDER']
        videos = []
        
        if os.path.exists(cache_folder):
            for filename in os.listdir(cache_folder):
                if filename.endswith('.json'):
                    cache_path = os.path.join(cache_folder, filename)
                    try:
                        with open(cache_path, 'r') as f:
                            cached_data = json.load(f)
                            
                        video_hash = filename.replace('.json', '')
                        video_filename = cached_data.get('video', 'Unknown')
                        video_info = {
                            "hash": video_hash,
                            "filename": video_filename,
                            "event_count": len(cached_data.get('events', [])),
                            "has_statistics": cached_data.get('statistics', {}).get('metadata', {}).get('processed', False),
                            "events": cached_data.get('events', []),
                            "video_url": f"/api/video/{video_filename}" if video_filename != 'Unknown' else None
                        }
                        
                        # Check if video file actually exists
                        if video_filename != 'Unknown':
                            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
                            video_info["video_exists"] = os.path.exists(video_path)
                        else:
                            video_info["video_exists"] = False
                        
                        # Get file modification time
                        mtime = os.path.getmtime(cache_path)
                        video_info["analyzed_at"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                        
                        videos.append(video_info)
                    except Exception as e:
                        print(f"⚠️ Error reading cache file {filename}: {e}")
                        continue
        
        # Sort by most recently analyzed first
        videos.sort(key=lambda x: x.get('analyzed_at', ''), reverse=True)
        
        return jsonify({
            "success": True,
            "videos": videos,
            "count": len(videos)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def generate_supercut_with_gemini(video_path, query, events):
    """Use Gemini AI to analyze query and select relevant events for supercut"""
    print(f"🎬 Generating supercut for query: '{query}'")
    
    # Convert events to a format Gemini can understand
    events_summary = []
    for event in events:
        events_summary.append({
            "time": event.get('time', '00:00'),
            "title": event.get('title', ''),
            "desc": event.get('desc', ''),
            "type": event.get('type', 'NORMAL')
        })
    
    prompt = f"""You are an NFL video editor creating a supercut based on a user's question.

USER QUESTION: "{query}"

AVAILABLE EVENTS IN THIS VIDEO:
{json.dumps(events_summary, indent=2)}

Your task:
1. Analyze the user's question and select the most relevant events
2. Create a compelling title for this supercut (max 8 words)
3. Write a brief intro explaining why these clips were selected (2-3 sentences)
4. Return ONLY valid JSON with this structure:

{{
  "title": "Compelling Supercut Title",
  "intro": "Brief explanation of why these clips were selected and what they show.",
  "selected_events": [
    {{
      "time": "MM:SS",
      "reason": "Why this clip is relevant to the question"
    }}
  ]
}}

IMPORTANT:
- Select 3-8 events that best answer the user's question
- Events are already analyzed and tagged - use their existing titles/descriptions
- The "time" field must match exactly one of the events above (MM:SS format)
- Be selective - quality over quantity
- The intro should explain the selection criteria

Return ONLY valid JSON, no markdown, no code blocks."""
    
    try:
        # Use Gemini to analyze and select events
        models_to_try = [
            'gemini-2.5-pro',
            'gemini-2.5-flash',
            'gemini-1.5-pro',
            'gemini-1.5-flash'
        ]
        
        last_error = None
        for model_name in models_to_try:
            try:
                print(f"🔄 Trying model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={"timeout": 60})
                print(f"✅ Successfully using model: {model_name}")
                break
            except Exception as model_error:
                last_error = model_error
                error_msg = str(model_error)
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(f"⚠️ Quota exceeded for {model_name}, trying next...")
                    continue
                print(f"⚠️ Error with {model_name}: {error_msg[:100]}... trying next...")
                continue
        else:
            if last_error:
                raise ValueError(f"All models failed. Last error: {str(last_error)}")
            else:
                raise ValueError("Failed to initialize any model")
        
        # Parse response
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        result = json.loads(text)
        
        # Match selected events with actual events and create clips
        clips = []
        events_by_time = {event.get('time'): event for event in events}
        
        for selected in result.get('selected_events', []):
            event_time = selected.get('time')
            if event_time in events_by_time:
                event = events_by_time[event_time]
                # Convert MM:SS to seconds
                time_parts = event_time.split(':')
                start_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                # Clip range: -2 seconds before, +4 seconds after
                clips.append({
                    "start": max(0, start_seconds - 2),
                    "end": start_seconds + 4,
                    "event": event,
                    "reason": selected.get('reason', '')
                })
        
        return {
            "title": result.get('title', 'Supercut'),
            "intro": result.get('intro', 'Selected clips from the video.'),
            "clips": clips,
            "query": query
        }
        
    except Exception as e:
        print(f"❌ Error generating supercut: {e}")
        raise


@app.route('/api/supercut', methods=['POST'])
def create_supercut():
    """Generate a supercut based on user query"""
    if 'query' not in request.form:
        return jsonify({"error": "No query provided"}), 400
    
    query = request.form['query'].strip()
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    # Get video file - either from upload, filename, or video_hash
    filename = None
    filepath = None
    video_hash = None
    
    # Check if video_hash is provided directly (from cached video list)
    if 'video_hash' in request.form:
        video_hash = request.form['video_hash']
        # Validate hash format
        if not re.match(r'^[a-f0-9]+$', video_hash):
            return jsonify({"error": "Invalid video hash format"}), 400
        # Try to load cache directly
        cached_data = load_cached_analysis(video_hash)
        if cached_data:
            filename = cached_data.get('video', 'Unknown')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) if filename != 'Unknown' else None
    elif 'video' in request.files and request.files['video'].filename:
        # New upload
        file = request.files['video']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    elif 'video_filename' in request.form:
        # Use existing video file
        filename = secure_filename(request.form['video_filename'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "Video file not found"}), 404
    else:
        return jsonify({"error": "No video file, filename, or video_hash provided"}), 400
    
    try:
        # Get video hash if not already provided
        if not video_hash:
            if filepath and os.path.exists(filepath):
                video_hash = get_video_hash(filepath)
            else:
                return jsonify({
                    "error": "Could not determine video hash. Please ensure the video file exists."
                }), 400
        
        # Load cached analysis
        if not cached_data:
            cached_data = load_cached_analysis(video_hash)
        
        if not cached_data or not cached_data.get('events'):
            # Debug: list available cache files
            cache_folder = app.config['CACHE_FOLDER']
            available_caches = []
            if os.path.exists(cache_folder):
                available_caches = [f for f in os.listdir(cache_folder) if f.endswith('.json')]
            
            return jsonify({
                "error": "Video must be analyzed first. Please analyze the video before creating a supercut.",
                "debug": {
                    "video_hash": video_hash,
                    "filename": filename,
                    "filepath_exists": filepath and os.path.exists(filepath) if filepath else False,
                    "cache_files_count": len(available_caches)
                }
            }), 400
        
        events = cached_data.get('events', [])
        
        # Generate supercut using Gemini
        supercut_data = generate_supercut_with_gemini(filepath, query, events)
        
        # Add metadata
        supercut_data['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        supercut_data['video_hash'] = video_hash
        supercut_data['video_filename'] = filename
        
        # Generate unique ID for sharing
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        supercut_id = f"{video_hash}_{query_hash}"
        supercut_data['id'] = supercut_id
        
        # Save supercut metadata
        supercut_filename = f"{supercut_id}.json"
        supercut_path = os.path.join(app.config['EDITS_FOLDER'], supercut_filename)
        
        with cache_lock:
            with open(supercut_path, 'w') as f:
                json.dump(supercut_data, f, indent=2)
        
        print(f"💾 Saved supercut: {supercut_filename} (ID: {supercut_id})")
        
        return jsonify({
            "success": True,
            "supercut": supercut_data
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error creating supercut: {error_msg}")
        return jsonify({
            "error": error_msg
        }), 500


@app.route('/api/cache/<video_hash>', methods=['GET'])
def get_cache(video_hash):
    """Get cached analysis by video hash"""
    try:
        # Security: ensure video_hash is safe
        if not re.match(r'^[a-f0-9]+$', video_hash):
            return jsonify({"error": "Invalid video hash"}), 400
        
        cached_data = load_cached_analysis(video_hash)
        
        if not cached_data:
            return jsonify({"error": "Cache not found"}), 404
        
        return jsonify({
            "success": True,
            "events": cached_data.get('events', []),
            "video": cached_data.get('video', 'Unknown')
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/supercut/<supercut_id>', methods=['GET'])
def get_supercut(supercut_id):
    """Get a supercut by ID (for sharing)"""
    try:
        # Security: ensure supercut_id is safe (alphanumeric and underscores only)
        if not re.match(r'^[a-f0-9_]+$', supercut_id):
            return jsonify({"error": "Invalid supercut ID"}), 400
        
        supercut_filename = f"{supercut_id}.json"
        supercut_path = os.path.join(app.config['EDITS_FOLDER'], supercut_filename)
        
        if not os.path.exists(supercut_path):
            return jsonify({"error": "Supercut not found"}), 404
        
        with open(supercut_path, 'r') as f:
            supercut_data = json.load(f)
        
        # Also include video URL if video exists
        video_filename = supercut_data.get('video_filename')
        if video_filename:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            if os.path.exists(video_path):
                supercut_data['video_url'] = f"/api/video/{video_filename}"
                supercut_data['video_exists'] = True
            else:
                supercut_data['video_exists'] = False
        
        return jsonify({
            "success": True,
            "supercut": supercut_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/analyze', methods=['GET', 'POST'])
def analyze_video():
    if request.method == 'GET':
        return jsonify({
            "error": "Method not allowed",
            "message": "This endpoint only accepts POST requests",
            "usage": {
                "method": "POST",
                "url": "/analyze",
                "content_type": "multipart/form-data",
                "field": "video",
                "example_curl": "curl -X POST -F 'video=@your_video.mp4' http://localhost:5001/analyze"
            }
        }), 405
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['video']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Check cache first
        video_hash = get_video_hash(filepath)
        cached_data = load_cached_analysis(video_hash)
        
        if cached_data:
            # Check if statistics exist
            has_statistics = cached_data.get('statistics') is not None and \
                           cached_data.get('statistics', {}).get('metadata', {}).get('processed', False)
            
            # Return cached analysis (with or without statistics)
            response_data = {
                "success": True,
                "events": cached_data.get('events', []),
                "count": len(cached_data.get('events', [])),
                "video": filename,
                "video_hash": video_hash,
                "cached": True
            }
            
            # Include statistics if available
            if has_statistics:
                response_data["statistics"] = cached_data.get('statistics')
                response_data["has_statistics"] = True
            else:
                response_data["has_statistics"] = False
                # Trigger background processing if statistics don't exist
                process_statistics_background(filepath, video_hash)
            
            return jsonify(response_data)
        
        # Pass A: Local Vision
        detector = NFLEventDetector(filepath)
        detector.detect_audio_events()
        detector.detect_visual_events()
        manifest = detector.generate_manifest()
        
        # Fallback if vision finds nothing (force generic analysis)
        if not manifest:
            manifest = [{"seconds": 10, "type": "sample_check"}, {"seconds": 30, "type": "sample_check"}]


        # Pass B: Gemini
        final_events = analyze_with_gemini(filepath, manifest)
        
        # Log stats info for debugging
        stats_count = sum(1 for event in final_events if event.get('stats'))
        print(f"📊 Events with stats: {stats_count}/{len(final_events)}")
        if stats_count > 0:
            sample_stats = next((e.get('stats') for e in final_events if e.get('stats')), {})
            print(f"📊 Sample stats keys: {list(sample_stats.keys())}")
        
        # Cache the analysis (without statistics initially)
        save_analysis_cache(video_hash, {"events": final_events, "video": filename})
        
        # Trigger background statistics processing
        process_statistics_background(filepath, video_hash)
        
        # Return JSON response for frontend
        # Response format: Array of event objects
        # Each event has: { "time": "MM:SS", "title": "...", "desc": "...", "type": "TOUCHDOWN|PENALTY|TIMEOUT|TURNOVER|NORMAL" }
        return jsonify({
            "success": True,
            "events": final_events,
            "count": len(final_events),
            "video": filename,
            "video_hash": video_hash,
            "cached": False,
            "has_statistics": False,
            "statistics_processing": "started"
        })


    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        # Check for API key errors
        if "API key" in error_msg.lower() or "API_KEY" in error_msg or "expired" in error_msg.lower():
            return jsonify({
                "error": "API Key Error",
                "message": "Your Gemini API key is invalid or expired.",
                "solution": "Please get a new API key from https://makersuite.google.com/app/apikey",
                "instructions": [
                    "1. Visit https://makersuite.google.com/app/apikey",
                    "2. Sign in with your Google account",
                    "3. Create a new API key or regenerate an existing one",
                    "4. Update your .env file with: API_KEY=your_new_key_here",
                    "5. Restart the server"
                ],
                "original_error": error_msg
            }), 401
        
        # Check for quota errors
        if "429" in error_msg or "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
            return jsonify({
                "error": "Quota Exceeded",
                "message": "You have exceeded your API quota for the requested model.",
                "solution": "The system will automatically try free-tier models. If this persists:",
                "instructions": [
                    "1. Check your usage at: https://ai.dev/usage?tab=rate-limit",
                    "2. Review rate limits: https://ai.google.dev/gemini-api/docs/rate-limits",
                    "3. Consider upgrading your plan or waiting for quota reset",
                    "4. The system will automatically try gemini-1.5-flash (free tier) first"
                ],
                "original_error": error_msg
            }), 429
        
        return jsonify({"error": error_msg}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)

