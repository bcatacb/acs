import os
import json
import logging
from typing import Optional
from emergentintegrations.llm.chat import LlmChat, UserMessage, FileContentWithMimeType

logger = logging.getLogger(__name__)

EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY")

ANALYSIS_SYSTEM_PROMPT = """You are an expert music producer and audio analyst specializing in hip-hop and rap music.
Your task is to analyze acapella rap recordings and provide detailed analysis for beat matching.

Analyze the following aspects:
1. BPM (beats per minute) - estimate based on the flow and cadence
2. Flow style - choppy, smooth, melodic, aggressive, etc.
3. Cadence - fast, medium, slow, variable
4. Mood - aggressive, chill, emotional, energetic, dark, uplifting
5. Lyric density - sparse, moderate, dense (how many words per bar)
6. Suggested genres - based on the flow, suggest 2-3 beat genres that would match

Return your analysis as a JSON object with these exact keys:
{
    "bpm": <integer between 60-200>,
    "flow_style": "<string>",
    "cadence": "<string>",
    "mood": "<string>",
    "lyric_density": "<string>",
    "suggested_genres": ["<genre1>", "<genre2>", "<genre3>"],
    "transcript": "<optional partial transcript of the lyrics>"
}

Only return the JSON object, no other text."""

async def analyze_acapella(audio_file_path: str, session_id: str) -> dict:
    """Analyze an acapella recording using Gemini's audio capabilities."""
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=session_id,
            system_message=ANALYSIS_SYSTEM_PROMPT
        ).with_model("gemini", "gemini-2.5-pro-preview-05-06")
        
        # Create file content for audio
        audio_file = FileContentWithMimeType(
            file_path=audio_file_path,
            mime_type="audio/webm"  # Browser recordings are typically webm
        )
        
        user_message = UserMessage(
            text="Analyze this acapella rap recording and provide the JSON analysis.",
            file_contents=[audio_file]
        )
        
        response = await chat.send_message(user_message)
        
        # Parse the JSON response
        # Clean up response if it has markdown code blocks
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        analysis = json.loads(response_text.strip())
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse analysis JSON: {e}")
        # Return default analysis if parsing fails
        return {
            "success": True,
            "analysis": {
                "bpm": 90,
                "flow_style": "smooth",
                "cadence": "medium",
                "mood": "chill",
                "lyric_density": "moderate",
                "suggested_genres": ["trap", "lo_fi", "boom_bap"],
                "transcript": None
            }
        }
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
