import os
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUNO_API_KEY = os.environ.get("SUNO_API_KEY")
SUNO_BASE_URL = "https://api.sunoapi.org/api/v1"

GENRE_PROMPTS = {
    "trap": "trap beat, heavy 808 bass, hi-hat rolls, dark synths, hard-hitting drums, modern hip-hop production",
    "boom_bap": "boom bap hip-hop beat, classic drums, vinyl samples, dusty soul chops, old school rap instrumental",
    "drill": "UK drill beat, sliding 808s, dark piano, aggressive drums, menacing synths, trap hi-hats",
    "lo_fi": "lo-fi hip-hop beat, jazzy chords, vinyl crackle, mellow drums, chill vibes, relaxing atmosphere",
    "west_coast": "west coast g-funk beat, bouncy synths, funk bass, smooth melodies, classic hip-hop instrumental",
    "east_coast": "east coast hip-hop beat, hard drums, piano samples, boom bap influence, raw sound",
    "southern": "southern hip-hop beat, crunk drums, heavy bass, bounce rhythm, club ready production",
    "melodic": "melodic rap beat, emotional piano, ambient pads, trap drums, atmospheric production"
}

async def generate_beat(genre: str, bpm: int, mood: str, flow_style: str) -> dict:
    """Generate a beat using Suno API based on analysis."""
    try:
        if not SUNO_API_KEY:
            return {"success": False, "error": "Suno API key not configured"}
        
        # Build the prompt based on analysis
        base_prompt = GENRE_PROMPTS.get(genre, GENRE_PROMPTS["trap"])
        prompt = f"{base_prompt}, {bpm} BPM, {mood} mood, matching {flow_style} rap flow, instrumental only, no vocals"
        
        headers = {
            "Authorization": f"Bearer {SUNO_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "customMode": False,
            "instrumental": True,
            "model": "V4_5ALL",
            "callBackUrl": "https://webhook.site/placeholder"
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{SUNO_BASE_URL}/generate",
                headers=headers,
                json=payload
            )
            
            data = response.json()
            
            if data.get("code") == 200:
                task_id = data.get("data", {}).get("taskId")
                return {
                    "success": True,
                    "task_id": task_id,
                    "status": "processing"
                }
            else:
                return {
                    "success": False,
                    "error": data.get("msg", "Unknown error")
                }
                
    except Exception as e:
        logger.error(f"Beat generation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def check_beat_status(task_id: str) -> dict:
    """Check the status of a beat generation task."""
    try:
        if not SUNO_API_KEY:
            return {"success": False, "error": "Suno API key not configured"}
        
        headers = {
            "Authorization": f"Bearer {SUNO_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{SUNO_BASE_URL}/generate/record-info?taskId={task_id}",
                headers=headers
            )
            
            # Handle 404 during processing
            if response.status_code == 404:
                return {
                    "success": True,
                    "status": "processing",
                    "task_id": task_id
                }
            
            data = response.json()
            
            if data.get("code") == 200:
                task_data = data.get("data", {})
                status = task_data.get("status", "processing")
                
                if status == "FIRST_SUCCESS" or status == "SUCCESS":
                    suno_data = task_data.get("response", {}).get("sunoData", [])
                    if suno_data:
                        beat = suno_data[0]
                        return {
                            "success": True,
                            "status": "complete",
                            "task_id": task_id,
                            "audio_url": beat.get("audioUrl"),
                            "image_url": beat.get("imageUrl"),
                            "title": beat.get("title"),
                            "duration": beat.get("duration")
                        }
                
                return {
                    "success": True,
                    "status": "processing",
                    "task_id": task_id
                }
            else:
                return {
                    "success": False,
                    "error": data.get("msg", "Unknown error")
                }
                
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
