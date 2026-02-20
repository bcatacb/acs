from fastapi import APIRouter, Query, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import io
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
import shutil
import tempfile

# It's better to use the provider and database classes directly
from soundbank.provider import SoundBankProvider
from soundbank.database import SoundAsset
from soundbank.ingest import ingest_samples
from soundbank.transformations import Transformations

# Setup logging
logger = logging.getLogger(__name__)


# Pydantic response model for API
class SoundAssetResponse(BaseModel):
    id: int
    original_filename: str
    start_sample: int
    end_sample: int
    category: str
    intensity_score: float
    spectral_hole: float
    rms_energy: float
    mid_range_density: float
    sample_rate: int
    duration_samples: int
    duration_seconds: float
    
    @classmethod
    def from_asset(cls, asset: SoundAsset) -> "SoundAssetResponse":
        """Convert SoundAsset dataclass to response model."""
        return cls(
            id=asset.id,
            original_filename=asset.original_filename,
            start_sample=asset.start_sample,
            end_sample=asset.end_sample,
            category=asset.category,
            intensity_score=asset.intensity_score,
            spectral_hole=asset.spectral_hole,
            rms_energy=asset.rms_energy,
            mid_range_density=asset.mid_range_density,
            sample_rate=asset.sample_rate,
            duration_samples=asset.duration_samples,
            duration_seconds=asset.duration_seconds
        )


class SpectralAnalysisResponse(BaseModel):
    has_notch: bool
    notch_depth_db: float
    notch_center_freq: float
    notch_bandwidth: float
    full_band_energy: float
    notch_band_energy: float
    confidence: float

soundbank_router = APIRouter(prefix="/api/soundbank")

# Base directory for soundbanks, relative to the backend folder
SOUNDBANK_BASE_DIR = Path(__file__).parent / "output"
Path(SOUNDBANK_BASE_DIR).mkdir(exist_ok=True)

def get_provider(soundbank_name: str) -> SoundBankProvider:
    """Helper to get a soundbank provider."""
    soundbank_path = (SOUNDBANK_BASE_DIR / soundbank_name).resolve()
    if not soundbank_path.is_dir() or not (SOUNDBANK_BASE_DIR in soundbank_path.parents):
         raise ValueError("Invalid soundbank name")
    
    master_wav = soundbank_path / "master_bank.wav"
    db_path = soundbank_path / "bank.db"
    
    if not master_wav.exists() or not db_path.exists():
        raise FileNotFoundError(f"Soundbank '{soundbank_name}' not found or is incomplete.")

    return SoundBankProvider(str(master_wav), str(db_path))

@soundbank_router.get("/list")
async def list_soundbanks():
    """
    List all available soundbanks.
    """
    soundbanks = [d.name for d in SOUNDBANK_BASE_DIR.iterdir() if d.is_dir()]
    return {"soundbanks": soundbanks}

@soundbank_router.get("/info")
async def get_soundbank_info(soundbank_name: str = "default"):
    """
    Get statistics and metadata for a given soundbank.
    """
    try:
        provider = get_provider(soundbank_name)
        return provider.get_statistics()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

@soundbank_router.get("/query", response_model=List[SoundAssetResponse])
async def query_assets(
    soundbank_name: str = "default",
    category: Optional[str] = None,
    intensity_min: Optional[float] = Query(None, ge=0.0, le=1.0),
    intensity_max: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    """
    Query assets in the soundbank, with optional filters.
    """
    try:
        provider = get_provider(soundbank_name)
        assets = provider.list_assets(category=category)

        # Filter by intensity if provided
        if intensity_min is not None:
            assets = [a for a in assets if provider.intensity_range.normalize(a.intensity_score) >= intensity_min]
        if intensity_max is not None:
            assets = [a for a in assets if provider.intensity_range.normalize(a.intensity_score) <= intensity_max]
        
        # Convert to response models
        return [SoundAssetResponse.from_asset(asset) for asset in assets]
    except (FileNotFoundError, ValueError):
        return []


@soundbank_router.get("/sample/{soundbank_name}/{asset_id}")
async def get_sample_audio(soundbank_name: str, asset_id: int):
    """
    Get the audio data for a specific asset as a WAV file.
    """
    try:
        provider = get_provider(soundbank_name)
        audio_data = provider.get_audio_by_id(asset_id)
        
        # Use an in-memory buffer to save the WAV file
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, provider.sample_rate, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")

    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
        
class IngestRequest(BaseModel):
    output_dir_name: str = "default"
    category: str
    # Add other options from ingest_samples as needed

async def _ingest_from_uploaded_files(
    files: List[UploadFile],
    output_path: Path,
    category: str,
    temp_dir: Path
):
    """
    Helper function to handle file ingestion from uploaded files.
    Runs in background task.
    
    The frontend sends files with webkitRelativePath as the filename,
    preserving directory structure.
    """
    try:
        # Save uploaded files to temp directory, preserving directory structure
        for file in files:
            # The filename contains the webkitRelativePath from the frontend
            # e.g., "Drums/Kicks/kick_001.wav"
            file_path = file.filename
            
            # Create subdirectories as needed
            full_path = temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            with open(full_path, 'wb') as f:
                content = await file.read()
                f.write(content)
        
        # Ingest from temp directory
        ingest_samples(
            input_dir=str(temp_dir),
            output_dir=str(output_path),
            category=category,
            recursive=True
        )
        
        logger.info(f"Successfully ingested {len(files)} files into soundbank '{output_path.name}'")
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            shutil.rmtree(temp_dir)

@soundbank_router.post("/ingest")
async def start_ingestion(
    output_dir_name: str = Form("default"),
    category: str = Form("loops"),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Start a background task to ingest uploaded audio files into a soundbank.
    
    Users can select any folder and the frontend will upload all files within it.
    The backend ingests them and stores the soundbank in output/{output_dir_name}/.
    """
    try:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Create output directory for soundbank
        output_path = SOUNDBANK_BASE_DIR / output_dir_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create temp directory to hold uploaded files
        temp_dir = Path(tempfile.mkdtemp(prefix="ingest_"))
        
        if background_tasks:
            background_tasks.add_task(
                _ingest_from_uploaded_files,
                files=files,
                output_path=output_path,
                category=category,
                temp_dir=temp_dir
            )
            return {
                "message": f"Ingestion started for {len(files)} file(s) into soundbank '{output_dir_name}'.",
                "soundbank_name": output_dir_name,
                "files_count": len(files)
            }
        else:
            # Fallback: synchronous ingestion
            await _ingest_from_uploaded_files(files, output_path, category, temp_dir)
            return {
                "message": f"Successfully ingested {len(files)} file(s) into soundbank '{output_dir_name}'.",
                "soundbank_name": output_dir_name,
                "files_count": len(files)
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

class TransformRequest(BaseModel):
    soundbank_name: str
    asset_id: int
    transformation: str
    params: dict = {}

@soundbank_router.post("/transform")
async def transform_asset(request: TransformRequest):
    """
    Apply a transformation to an asset and return the result.
    """
    try:
        provider = get_provider(request.soundbank_name)
        audio = provider.get_audio_by_id(request.asset_id)
        
        transformed_audio = None
        if request.transformation == 'notch':
            transformed_audio, _ = Transformations.apply_spectral_notch(audio, provider.sample_rate)
        elif request.transformation == 'bitcrush':
            transformed_audio = Transformations.bit_crush(audio, **request.params)
        elif request.transformation == 'stretch':
            transformed_audio = Transformations.time_stretch(audio, provider.sample_rate, **request.params)
        else:
            raise HTTPException(status_code=400, detail="Unknown transformation")
            
        buffer = io.BytesIO()
        sf.write(buffer, transformed_audio, provider.sample_rate, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")

    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@soundbank_router.post("/replace-asset")
async def replace_asset(
    soundbank_name: str = Query(...),
    asset_id: int = Query(...),
    file: UploadFile = File(...)
):
    """
    Replace an asset in the soundbank with new audio data.
    
    The new audio will be written to the master WAV file and the database
    will be updated with new metadata (RMS energy, duration, etc).
    """
    try:
        logger.info(f"Starting asset replacement: soundbank={soundbank_name}, asset_id={asset_id}")
        
        soundbank_path = (SOUNDBANK_BASE_DIR / soundbank_name).resolve()
        master_wav_path = soundbank_path / "master_bank.wav"
        db_path = soundbank_path / "bank.db"
        
        if not master_wav_path.exists() or not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Soundbank '{soundbank_name}' not found")
        
        # Read uploaded audio
        file_content = await file.read()
        audio_buffer = io.BytesIO(file_content)
        new_audio, sample_rate = sf.read(audio_buffer)
        
        # Convert to mono if needed
        if new_audio.ndim > 1:
            new_audio = np.mean(new_audio, axis=1)
        
        logger.info(f"Loaded new audio: {len(new_audio)} samples at {sample_rate}Hz")
        
        # Get current asset and provider
        provider = get_provider(soundbank_name)
        old_asset = provider.db.get_asset_by_id(asset_id)
        
        if not old_asset:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")
        
        logger.info(f"Old asset: {old_asset.duration_samples} samples, {old_asset.start_sample}-{old_asset.end_sample}")
        
        # Load current master WAV
        master_audio, master_sr = sf.read(str(master_wav_path))
        
        # Extract audio before and after the old asset
        audio_before = master_audio[:old_asset.start_sample]
        audio_after = master_audio[old_asset.end_sample:]
        
        # Ensure new audio matches sample rate
        if sample_rate != provider.sample_rate:
            new_audio = Transformations.resample(new_audio, sample_rate, provider.sample_rate)
            logger.info(f"Resampled new audio to {provider.sample_rate}Hz")
        
        # Concatenate new master audio
        new_master_audio = np.concatenate([audio_before, new_audio.astype(np.float32), audio_after])
        
        # Update database with new asset information
        new_start = len(audio_before)
        new_end = new_start + len(new_audio)
        new_duration = len(new_audio)
        
        # Calculate new metadata
        new_rms = Transformations.calculate_rms_energy(new_audio)
        new_mid_range = Transformations.calculate_mid_range_density(new_audio, provider.sample_rate)
        
        with provider.db:
            # Update the asset record
            provider.db.conn.execute("""
                UPDATE assets 
                SET start_sample = ?, end_sample = ?, duration_samples = ?,
                    rms_energy = ?, intensity_score = ?, mid_range_density = ?
                WHERE id = ?
            """, (new_start, new_end, new_duration, new_rms, new_rms, new_mid_range, asset_id))
            
            # Update all assets that come after this one
            later_assets = provider.db.conn.execute("""
                SELECT id, start_sample, end_sample, duration_samples 
                FROM assets 
                WHERE start_sample >= ?
                ORDER BY start_sample
            """, (old_asset.end_sample,)).fetchall()
            
            offset = len(new_audio) - old_asset.duration_samples
            logger.info(f"Offset for later assets: {offset} samples")
            
            for asset_row in later_assets:
                asset_id_row, start, end, dur = asset_row
                if asset_id_row != asset_id:  # Skip the one we just updated
                    new_asset_start = start + offset
                    new_asset_end = end + offset
                    provider.db.conn.execute("""
                        UPDATE assets 
                        SET start_sample = ?, end_sample = ?
                        WHERE id = ?
                    """, (new_asset_start, new_asset_end, asset_id_row))
                    logger.info(f"Updated asset {asset_id_row}: {new_asset_start}-{new_asset_end}")
            
            provider.db.conn.commit()
        
        # Write new master WAV
        sf.write(str(master_wav_path), new_master_audio, provider.sample_rate, subtype='PCM_24')
        logger.info(f"Wrote master WAV: {len(new_master_audio)} samples")
        
        logger.info(f"Asset replacement complete for asset {asset_id}")
        return {
            "success": True,
            "message": f"Asset {asset_id} replaced successfully",
            "new_duration_samples": new_duration,
            "new_duration_seconds": new_duration / provider.sample_rate,
            "new_rms_energy": float(new_rms)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Asset replacement failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Asset replacement failed: {str(e)}")

# ... (add transform endpoint later)

class VerifyRequest(BaseModel):
    soundbank_name: str

@soundbank_router.post("/verify")
async def verify_soundbank(request: VerifyRequest):
    """
    Verify the sample accuracy of a soundbank.
    """
    try:
        from soundbank.test_generator import TestSampleGenerator
        
        soundbank_path = (SOUNDBANK_BASE_DIR / request.soundbank_name).resolve()
        master_wav = soundbank_path / "master_bank.wav"
        db_path = soundbank_path / "bank.db"
        
        if not master_wav.exists() or not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Soundbank '{request.soundbank_name}' not found or is incomplete.")
            
        generator = TestSampleGenerator()
        # Use non-strict verification for real audio files
        passed, errors = generator.verify_sample_accuracy(
            str(master_wav), str(db_path), strict_markers=False
        )
        
        return {"passed": passed, "errors": errors}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SpectralRequest(BaseModel):
    soundbank_name: str
    asset_id: Optional[int] = None

@soundbank_router.post("/spectral")
async def spectral_verification(request: SpectralRequest):
    """
    Verify the spectral notch of assets in a soundbank.
    
    Returns a summary of spectral analysis results.
    Note: This can take a while for large soundbanks as it analyzes each asset.
    """
    try:
        logger.info(f"Starting spectral verification for soundbank: {request.soundbank_name}")
        
        soundbank_path = (SOUNDBANK_BASE_DIR / request.soundbank_name).resolve()
        master_wav = soundbank_path / "master_bank.wav"
        db_path = soundbank_path / "bank.db"
        
        if not master_wav.exists() or not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Soundbank '{request.soundbank_name}' not found or incomplete")
        
        try:
            provider = get_provider(request.soundbank_name)
        except Exception as e:
            logger.error(f"Failed to load provider: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load soundbank: {str(e)}")
        
        logger.info(f"Provider loaded successfully")
        
        try:
            if request.asset_id:
                logger.info(f"Verifying single asset: {request.asset_id}")
                result = provider.verify_spectral_notch(asset_id=request.asset_id)
                logger.info(f"Spectral analysis complete for asset {request.asset_id}")
                return {
                    "asset_id": request.asset_id,
                    "has_notch": bool(result.has_notch),
                    "notch_depth_db": float(result.notch_depth_db),
                    "confidence": float(result.confidence)
                }
            else:
                logger.info(f"Verifying all assets in soundbank")
                assets = provider.list_assets()
                logger.info(f"Found {len(assets)} assets to analyze")
                
                # Return quick summary without doing all analysis
                return {
                    "soundbank": request.soundbank_name,
                    "total_assets": len(assets),
                    "message": "Use single asset_id parameter to analyze individual assets for detailed spectral results"
                }
        except Exception as e:
            logger.error(f"Spectral analysis error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Spectral analysis failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))