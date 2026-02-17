from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
import uuid
import aiofiles

# Load env before other imports
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

from models import (
    UserCreate, UserLogin, User, UserInDB, Token,
    ProjectCreate, ProjectUpdate, Project, AudioAnalysis, BeatGeneration,
    GENRES, GenreResponse
)
from auth import (
    verify_password, get_password_hash, create_access_token,
    get_current_user
)
from services.audio_analyzer import analyze_acapella
from services.beat_generator import generate_beat, check_beat_status
from services.accompaniment_generator import generate_true_accompaniment

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create uploads directory
UPLOADS_DIR = ROOT_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Create the main app
app = FastAPI(title="FlowState API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== Auth Routes ==============

@api_router.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check username
    existing_username = await db.users.find_one({"username": user_data.username})
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)
    
    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "username": user_data.username,
        "hashed_password": hashed_password,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.insert_one(user_doc)
    
    # Create token
    access_token = create_access_token(data={"sub": user_id, "email": user_data.email})
    
    user = User(id=user_id, email=user_data.email, username=user_data.username)
    return Token(access_token=access_token, user=user)

@api_router.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    user_doc = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not verify_password(user_data.password, user_doc["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token(data={"sub": user_doc["id"], "email": user_doc["email"]})
    
    user = User(
        id=user_doc["id"],
        email=user_doc["email"],
        username=user_doc["username"]
    )
    return Token(access_token=access_token, user=user)

@api_router.get("/auth/me", response_model=User)
async def get_me(current_user: dict = Depends(get_current_user)):
    user_doc = await db.users.find_one({"id": current_user["user_id"]}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    return User(
        id=user_doc["id"],
        email=user_doc["email"],
        username=user_doc["username"]
    )

# ============== Genre Routes ==============

@api_router.get("/genres", response_model=GenreResponse)
async def get_genres():
    return GenreResponse(genres=GENRES)

# ============== Project Routes ==============

@api_router.post("/projects", response_model=Project)
async def create_project(
    project_data: ProjectCreate,
    current_user: dict = Depends(get_current_user)
):
    project_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    
    project_doc = {
        "id": project_id,
        "user_id": current_user["user_id"],
        "name": project_data.name,
        "genre": project_data.genre,
        "instrument_pack": project_data.instrument_pack or "auto",
        "acapella_url": None,
        "analysis": None,
        "beat": None,
        "status": "draft",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }
    
    await db.projects.insert_one(project_doc)
    
    return Project(**{**project_doc, "created_at": now, "updated_at": now})

@api_router.get("/projects", response_model=List[Project])
async def get_projects(current_user: dict = Depends(get_current_user)):
    projects = await db.projects.find(
        {"user_id": current_user["user_id"]},
        {"_id": 0}
    ).sort("created_at", -1).to_list(100)
    
    # Convert timestamps
    for p in projects:
        if isinstance(p.get("created_at"), str):
            p["created_at"] = datetime.fromisoformat(p["created_at"])
        if isinstance(p.get("updated_at"), str):
            p["updated_at"] = datetime.fromisoformat(p["updated_at"])
    
    return projects

@api_router.get("/projects/{project_id}", response_model=Project)
async def get_project(project_id: str, current_user: dict = Depends(get_current_user)):
    project = await db.projects.find_one(
        {"id": project_id, "user_id": current_user["user_id"]},
        {"_id": 0}
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if isinstance(project.get("created_at"), str):
        project["created_at"] = datetime.fromisoformat(project["created_at"])
    if isinstance(project.get("updated_at"), str):
        project["updated_at"] = datetime.fromisoformat(project["updated_at"])
    
    return project

@api_router.patch("/projects/{project_id}", response_model=Project)
async def update_project(
    project_id: str,
    update_data: ProjectUpdate,
    current_user: dict = Depends(get_current_user)
):
    project = await db.projects.find_one(
        {"id": project_id, "user_id": current_user["user_id"]}
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
    update_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.projects.update_one({"id": project_id}, {"$set": update_dict})
    
    updated = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if isinstance(updated.get("created_at"), str):
        updated["created_at"] = datetime.fromisoformat(updated["created_at"])
    if isinstance(updated.get("updated_at"), str):
        updated["updated_at"] = datetime.fromisoformat(updated["updated_at"])
    
    return updated

@api_router.delete("/projects/{project_id}")
async def delete_project(project_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.projects.delete_one(
        {"id": project_id, "user_id": current_user["user_id"]}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted"}

# ============== Audio Upload & Analysis ==============

@api_router.post("/projects/{project_id}/upload")
async def upload_acapella(
    project_id: str,
    audio: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    project = await db.projects.find_one(
        {"id": project_id, "user_id": current_user["user_id"]}
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Save the audio file
    file_ext = audio.filename.split(".")[-1] if audio.filename else "webm"
    filename = f"{project_id}.{file_ext}"
    filepath = UPLOADS_DIR / filename
    
    async with aiofiles.open(filepath, "wb") as f:
        content = await audio.read()
        await f.write(content)
    
    # Update project with file path
    await db.projects.update_one(
        {"id": project_id},
        {"$set": {
            "acapella_url": str(filepath),
            "status": "uploaded",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    return {"message": "Audio uploaded", "filename": filename}

@api_router.post("/projects/{project_id}/analyze")
async def analyze_project(project_id: str, current_user: dict = Depends(get_current_user)):
    project = await db.projects.find_one(
        {"id": project_id, "user_id": current_user["user_id"]},
        {"_id": 0}
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not project.get("acapella_url"):
        raise HTTPException(status_code=400, detail="No audio uploaded")
    
    # Update status to analyzing
    await db.projects.update_one(
        {"id": project_id},
        {"$set": {"status": "analyzing", "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    # Analyze the audio
    result = await analyze_acapella(
        project["acapella_url"],
        session_id=f"analysis_{project_id}"
    )
    
    if result["success"]:
        analysis = result["analysis"]
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {
                "analysis": analysis,
                "status": "analyzed",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        return {"message": "Analysis complete", "analysis": analysis}
    else:
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {"status": "error", "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

# ============== Beat Generation ==============

@api_router.post("/projects/{project_id}/generate")
async def generate_project_beat(project_id: str, current_user: dict = Depends(get_current_user)):
    project = await db.projects.find_one(
        {"id": project_id, "user_id": current_user["user_id"]},
        {"_id": 0}
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not project.get("analysis"):
        raise HTTPException(status_code=400, detail="Project not analyzed yet")
    
    analysis = project["analysis"]
    genre = project.get("genre", "trap")
    instrument_pack = project.get("instrument_pack", "auto")
    
    # Update status to generating
    await db.projects.update_one(
        {"id": project_id},
        {"$set": {"status": "generating", "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    # Generate accompaniment locally using vocal timing/features.
    result = await generate_true_accompaniment(
        project_id=project_id,
        audio_file_path=project["acapella_url"],
        analysis=analysis,
        genre=genre,
        instrument_pack=instrument_pack,
    )
    
    if result["success"]:
        task_id = str(uuid.uuid4())
        beat_data = {
            "task_id": task_id,
            "status": "complete",
            "audio_url": None,  # populated dynamically in beat-status response
            "audio_path": result.get("audio_path"),
            "mix_url": None,    # populated dynamically in beat-status response
            "mix_path": result.get("mix_path"),
            "image_url": None,
            "title": result.get("title"),
            "duration": result.get("duration"),
            "bpm": result.get("bpm"),
            "onset_count": result.get("onset_count"),
            "repetition_score": result.get("repetition_score"),
            "render_attempts": result.get("render_attempts"),
            "instrument_palette": result.get("instrument_palette"),
            "instrument_pack": result.get("instrument_pack"),
            "instrument_system_version": result.get("instrument_system_version"),
            "loop_source": result.get("loop_source"),
            "arrangement_debug_path": result.get("arrangement_debug_path"),
            "midi_sources": result.get("midi_sources"),
            "asset_counts": result.get("asset_counts"),
        }
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {
                "beat": beat_data,
                "status": "complete",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        return {"message": "Accompaniment and mix generated", "task_id": task_id}
    else:
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {"status": "error", "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        raise HTTPException(status_code=500, detail=result.get("error", "Generation failed"))

@api_router.get("/projects/{project_id}/beat-status")
async def get_beat_status(project_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    project = await db.projects.find_one(
        {"id": project_id, "user_id": current_user["user_id"]},
        {"_id": 0}
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not project.get("beat") or not project["beat"].get("task_id"):
        raise HTTPException(status_code=400, detail="No beat generation in progress")
    
    beat = project["beat"]
    task_id = beat["task_id"]

    # Local accompaniment generation is synchronous: once stored, it is complete.
    if beat.get("status") == "complete":
        audio_url = str(request.base_url).rstrip("/") + f"/api/projects/{project_id}/beat-audio"
        mix_url = str(request.base_url).rstrip("/") + f"/api/projects/{project_id}/mix-audio"
        return {
            "success": True,
            "status": "complete",
            "task_id": task_id,
            "audio_url": audio_url,
            "mix_url": mix_url,
            "image_url": beat.get("image_url"),
            "title": beat.get("title"),
            "duration": beat.get("duration"),
            "bpm": beat.get("bpm"),
            "onset_count": beat.get("onset_count"),
            "repetition_score": beat.get("repetition_score"),
            "render_attempts": beat.get("render_attempts"),
            "instrument_palette": beat.get("instrument_palette"),
            "instrument_pack": beat.get("instrument_pack"),
            "instrument_system_version": beat.get("instrument_system_version"),
            "loop_source": beat.get("loop_source"),
            "arrangement_debug_path": beat.get("arrangement_debug_path"),
            "midi_sources": beat.get("midi_sources"),
            "asset_counts": beat.get("asset_counts"),
        }

    # Backward-compatible path for legacy remote generation records.
    result = await check_beat_status(task_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Status check failed"))

    if result["status"] == "complete":
        beat_data = {
            "task_id": task_id,
            "status": "complete",
            "audio_url": result.get("audio_url"),
            "image_url": result.get("image_url"),
            "title": result.get("title"),
            "duration": result.get("duration")
        }
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {
                "beat": beat_data,
                "status": "complete",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }}
        )
    return result


@api_router.get("/projects/{project_id}/beat-audio")
async def get_beat_audio(project_id: str, current_user: dict = Depends(get_current_user)):
    project = await db.projects.find_one(
        {"id": project_id, "user_id": current_user["user_id"]},
        {"_id": 0}
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    beat = project.get("beat") or {}
    audio_path = beat.get("audio_path")
    if not audio_path:
        raise HTTPException(status_code=404, detail="No local beat audio found")

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Beat audio file missing")

    return FileResponse(audio_path, media_type="audio/wav", filename=f"{project_id}_accompaniment.wav")


@api_router.get("/projects/{project_id}/mix-audio")
async def get_mix_audio(project_id: str, current_user: dict = Depends(get_current_user)):
    project = await db.projects.find_one(
        {"id": project_id, "user_id": current_user["user_id"]},
        {"_id": 0}
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    beat = project.get("beat") or {}
    mix_path = beat.get("mix_path")
    if not mix_path:
        raise HTTPException(status_code=404, detail="No local mix audio found")

    if not os.path.exists(mix_path):
        raise HTTPException(status_code=404, detail="Mix audio file missing")

    return FileResponse(mix_path, media_type="audio/wav", filename=f"{project_id}_mix.wav")

# ============== Health Check ==============

@api_router.get("/")
async def root():
    return {"message": "FlowState API", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy"}

# Include the router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
