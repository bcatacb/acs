# FlowState - Rap Beat Generation App PRD

## Original Problem Statement
Build a beats/instrumental making app for rap music similar to Suno, except instead of typing a prompt, users rap their song acapella to the app and it takes their performance and creates a beat tailored to their flow.

## User Personas
1. **Independent Rap Artists** - Musicians who want custom beats that match their unique flow
2. **Hip-Hop Content Creators** - YouTubers, TikTokers creating rap content
3. **Beat Makers** - Producers looking for AI-assisted beat creation tools
4. **Hobbyist Rappers** - Casual users exploring music creation

## Core Requirements (Static)
- Audio recording with waveform visualization
- AI-powered analysis of acapella recordings (BPM, flow, cadence, mood)
- Beat generation based on analyzed parameters
- Multiple genre support (trap, boom bap, drill, lo-fi, etc.)
- User authentication (JWT-based)
- Project management (create, edit, delete)
- Export/download capabilities

## What's Been Implemented (Jan 2026)
### Backend
- [x] FastAPI server with /api prefix routing
- [x] JWT authentication (register, login, me endpoints)
- [x] MongoDB integration for users and projects
- [x] Gemini AI integration for audio analysis
- [x] Suno API integration for beat generation
- [x] File upload handling for audio recordings
- [x] 8 genre options with descriptions

### Frontend
- [x] Landing page with hero section and features
- [x] Auth pages (login/register) with split layout
- [x] Dashboard with project cards and management
- [x] Studio page with:
  - Real-time waveform visualization
  - Audio recording via MediaRecorder API
  - Play/pause/re-record controls
  - Genre selection dropdown
  - Volume slider
  - Analyze Flow button (Gemini integration)
  - Generate Beat button (Suno integration)
  - Download buttons for acapella and generated beat
- [x] Protected routing
- [x] Dark cyber-noir theme (Chivo/Manrope/JetBrains Mono fonts)
- [x] Responsive design

## Prioritized Backlog

### P0 (Critical) - Completed
- ~~User auth~~ ✅
- ~~Project CRUD~~ ✅
- ~~Audio recording~~ ✅
- ~~Waveform visualization~~ ✅
- ~~Basic UI/UX~~ ✅

### P1 (Important)
- Audio file format support (MP3, WAV upload)
- Mix/combine acapella with generated beat
- Playback synchronization controls
- Project sharing/collaboration

### P2 (Nice to Have)
- Social login (Google/Apple)
- Public beat library
- Community features
- Beat customization post-generation
- Mobile app (React Native)

## Tech Stack
- **Frontend**: React 19, Tailwind CSS, Framer Motion, Shadcn/UI
- **Backend**: FastAPI, Motor (MongoDB async), Python 3.11
- **Database**: MongoDB
- **AI Services**: 
  - Gemini (audio analysis via Emergent LLM Key)
  - Suno API (beat generation)
- **Auth**: JWT with bcrypt password hashing

## API Keys Required
- EMERGENT_LLM_KEY (for Gemini)
- SUNO_API_KEY (for beat generation)

## Next Tasks
1. Add audio file upload support (upload existing recordings)
2. Implement beat+acapella mixing/preview
3. Add project duplication feature
4. Implement usage/credit tracking
5. Add more detailed analysis visualization
