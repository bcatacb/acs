import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
    Mic, ArrowLeft, Play, Pause, Square, Download, 
    Loader2, Zap, Music, RefreshCw, Volume2, Check, AlertCircle
} from 'lucide-react';
import { Button } from '../components/ui/button';
import { 
    Select, 
    SelectContent, 
    SelectItem, 
    SelectTrigger, 
    SelectValue 
} from '../components/ui/select';
import { Slider } from '../components/ui/slider';
import { useAuth } from '../context/AuthContext';
import { projectsApi, genresApi } from '../api/projects';
import { toast } from 'sonner';

const WaveformVisualizer = ({ audioData, isRecording, isPlaying }) => {
    const canvasRef = useRef(null);
    const animationRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        const draw = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fillRect(0, 0, width, height);

            const barCount = 64;
            const barWidth = (width / barCount) - 2;
            
            for (let i = 0; i < barCount; i++) {
                let barHeight;
                
                if (isRecording && audioData && audioData.length > 0) {
                    const dataIndex = Math.floor(i * audioData.length / barCount);
                    barHeight = (audioData[dataIndex] / 255) * height * 0.8;
                } else if (isPlaying) {
                    barHeight = Math.random() * height * 0.6 + 10;
                } else {
                    barHeight = Math.sin(i * 0.2 + Date.now() * 0.001) * 20 + 30;
                }

                const gradient = ctx.createLinearGradient(0, height - barHeight, 0, height);
                gradient.addColorStop(0, isRecording ? '#00F0FF' : '#7C3AED');
                gradient.addColorStop(1, isRecording ? '#7C3AED' : '#10B981');

                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.roundRect(
                    i * (barWidth + 2), 
                    height - barHeight, 
                    barWidth, 
                    barHeight,
                    [4, 4, 0, 0]
                );
                ctx.fill();
            }

            animationRef.current = requestAnimationFrame(draw);
        };

        draw();

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [audioData, isRecording, isPlaying]);

    return (
        <canvas 
            ref={canvasRef} 
            width={800} 
            height={160}
            className="w-full h-40 rounded-lg"
        />
    );
};

const INSTRUMENT_PACKS = [
    { id: 'auto', name: 'Auto' },
    { id: 'dark', name: 'Dark' },
    { id: 'warm', name: 'Warm' },
    { id: 'orchestral', name: 'Orchestral' },
];

export const StudioPage = () => {
    const { projectId } = useParams();
    const { token } = useAuth();
    const navigate = useNavigate();

    const [project, setProject] = useState(null);
    const [genres, setGenres] = useState([]);
    const [loading, setLoading] = useState(true);
    
    // Recording state
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const [audioBlob, setAudioBlob] = useState(null);
    const [audioUrl, setAudioUrl] = useState(null);
    const [audioData, setAudioData] = useState([]);
    
    // Playback state
    const [isPlaying, setIsPlaying] = useState(false);
    const [volume, setVolume] = useState(80);
    
    // Processing state
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [isGenerating, setIsGenerating] = useState(false);
    const pollIntervalRef = useRef(null);

    const mediaRecorderRef = useRef(null);
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const audioRef = useRef(null);
    const timerRef = useRef(null);
    const chunksRef = useRef([]);
    const beatObjectUrlRef = useRef(null);
    const mixObjectUrlRef = useRef(null);

    const revokeBeatObjectUrl = useCallback(() => {
        if (beatObjectUrlRef.current) {
            URL.revokeObjectURL(beatObjectUrlRef.current);
            beatObjectUrlRef.current = null;
        }
        if (mixObjectUrlRef.current) {
            URL.revokeObjectURL(mixObjectUrlRef.current);
            mixObjectUrlRef.current = null;
        }
    }, []);

    const loadProject = useCallback(async () => {
        try {
            const [projectData, genresData] = await Promise.all([
                projectsApi.getOne(token, projectId),
                genresApi.getAll()
            ]);
            setProject(projectData);
            setGenres(genresData);
        } catch (error) {
            toast.error('Failed to load project');
            navigate('/dashboard');
        } finally {
            setLoading(false);
        }
    }, [token, projectId, navigate]);

    useEffect(() => {
        loadProject();
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            revokeBeatObjectUrl();
        };
    }, [loadProject, revokeBeatObjectUrl]);

    const toAuthorizedBeat = useCallback(async (status) => {
        if (!status?.audio_url || !token) return status;
        const beatResponse = await fetch(status.audio_url, {
            headers: { Authorization: `Bearer ${token}` }
        });
        if (!beatResponse.ok) {
            throw new Error(`Failed to fetch beat audio (${beatResponse.status})`);
        }
        const beatBlob = await beatResponse.blob();
        revokeBeatObjectUrl();
        const beatObjectUrl = URL.createObjectURL(beatBlob);
        beatObjectUrlRef.current = beatObjectUrl;

        let mixObjectUrl = null;
        if (status?.mix_url) {
            const mixResponse = await fetch(status.mix_url, {
                headers: { Authorization: `Bearer ${token}` }
            });
            if (mixResponse.ok) {
                const mixBlob = await mixResponse.blob();
                mixObjectUrl = URL.createObjectURL(mixBlob);
                mixObjectUrlRef.current = mixObjectUrl;
            }
        }

        return { ...status, audio_url: beatObjectUrl, mix_url: mixObjectUrl };
    }, [token, revokeBeatObjectUrl]);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            analyserRef.current = audioContextRef.current.createAnalyser();
            const source = audioContextRef.current.createMediaStreamSource(stream);
            source.connect(analyserRef.current);
            analyserRef.current.fftSize = 256;

            mediaRecorderRef.current = new MediaRecorder(stream);
            chunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                setAudioBlob(blob);
                setAudioUrl(URL.createObjectURL(blob));
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorderRef.current.start(100);
            setIsRecording(true);
            setRecordingTime(0);

            // Start timer
            timerRef.current = setInterval(() => {
                setRecordingTime(prev => prev + 1);
            }, 1000);

            // Start audio visualization
            const updateAudioData = () => {
                if (!analyserRef.current || !isRecording) return;
                const data = new Uint8Array(analyserRef.current.frequencyBinCount);
                analyserRef.current.getByteFrequencyData(data);
                setAudioData(Array.from(data));
                if (isRecording) requestAnimationFrame(updateAudioData);
            };
            updateAudioData();

        } catch (error) {
            toast.error('Failed to access microphone');
            console.error(error);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            if (timerRef.current) clearInterval(timerRef.current);
        }
    };

    const handlePlayPause = () => {
        if (!audioRef.current) return;
        
        if (isPlaying) {
            audioRef.current.pause();
        } else {
            audioRef.current.play();
        }
        setIsPlaying(!isPlaying);
    };

    const handleVolumeChange = (value) => {
        setVolume(value[0]);
        if (audioRef.current) {
            audioRef.current.volume = value[0] / 100;
        }
    };

    const handleUploadAndAnalyze = async () => {
        if (!audioBlob) {
            toast.error('Please record your acapella first');
            return;
        }

        setIsAnalyzing(true);
        try {
            // Upload audio
            await projectsApi.uploadAudio(token, projectId, audioBlob);
            toast.success('Audio uploaded!');

            // Analyze
            const result = await projectsApi.analyze(token, projectId);
            setProject(prev => ({ ...prev, analysis: result.analysis, status: 'analyzed' }));
            toast.success('Analysis complete!');
        } catch (error) {
            toast.error(error.response?.data?.detail || 'Analysis failed');
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleGenerateBeat = async () => {
        if (!project?.analysis) {
            toast.error('Please analyze your recording first');
            return;
        }

        setIsGenerating(true);
        revokeBeatObjectUrl();
        try {
            const result = await projectsApi.generateBeat(token, projectId);
            setProject(prev => ({ 
                ...prev, 
                beat: { task_id: result.task_id, status: 'processing' },
                status: 'generating'
            }));
            toast.success('Accompaniment generation started!');

            // Try immediate status fetch first (local generation completes synchronously).
            try {
                const immediate = await projectsApi.checkBeatStatus(token, projectId);
                if (immediate.status === 'complete') {
                    const withAudio = await toAuthorizedBeat(immediate);
                    setProject(prev => ({
                        ...prev,
                        beat: withAudio,
                        status: 'complete'
                    }));
                    setIsGenerating(false);
                    toast.success('Accompaniment generated!');
                    return;
                }
            } catch (error) {
                console.error('Immediate status check failed:', error);
            }

            // Start polling for status
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            const interval = setInterval(async () => {
                try {
                    const status = await projectsApi.checkBeatStatus(token, projectId);
                    if (status.status === 'complete') {
                        const withAudio = await toAuthorizedBeat(status);
                        clearInterval(interval);
                        pollIntervalRef.current = null;
                        setProject(prev => ({
                            ...prev,
                            beat: withAudio,
                            status: 'complete'
                        }));
                        setIsGenerating(false);
                        toast.success('Accompaniment generated!');
                    }
                } catch (error) {
                    console.error('Status check failed:', error);
                }
            }, 10000);
            
            pollIntervalRef.current = interval;

        } catch (error) {
            toast.error(error.response?.data?.detail || 'Generation failed');
            setIsGenerating(false);
        }
    };

    const handleGenreChange = async (genre) => {
        try {
            await projectsApi.update(token, projectId, { genre });
            setProject(prev => ({ ...prev, genre }));
        } catch (error) {
            toast.error('Failed to update genre');
        }
    };

    const handleInstrumentPackChange = async (instrument_pack) => {
        try {
            await projectsApi.update(token, projectId, { instrument_pack });
            setProject(prev => ({ ...prev, instrument_pack }));
        } catch (error) {
            toast.error('Failed to update instrument pack');
        }
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-background flex items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background">
            {/* Noise overlay */}
            <div className="noise-overlay" />
            <div className="fixed inset-0 bg-gradient-radial pointer-events-none" />

            {/* Header */}
            <header className="relative z-10 flex items-center justify-between px-6 py-4 border-b border-border/40">
                <div className="flex items-center gap-4">
                    <Button 
                        variant="ghost" 
                        size="icon"
                        onClick={() => navigate('/dashboard')}
                        data-testid="back-btn"
                    >
                        <ArrowLeft className="w-5 h-5" />
                    </Button>
                    <div>
                        <h1 className="font-heading text-xl font-bold">{project?.name}</h1>
                        <span className="text-xs text-muted-foreground capitalize">
                            {project?.genre?.replace('_', ' ')}
                        </span>
                    </div>
                </div>
                <Link to="/" className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                        <Mic className="w-4 h-4 text-white" />
                    </div>
                    <span className="font-heading font-bold hidden md:block">FlowState</span>
                </Link>
            </header>

            {/* Main Content */}
            <main className="relative z-10 p-6 max-w-6xl mx-auto">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    
                    {/* Recording Section */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Waveform */}
                        <div className="bg-card border border-border/40 rounded-xl p-6">
                            <div className="flex items-center justify-between mb-4">
                                <h2 className="font-heading text-lg font-semibold">Recording Studio</h2>
                                <span className="font-mono text-sm text-muted-foreground">
                                    {formatTime(recordingTime)}
                                </span>
                            </div>
                            
                            <div className="waveform-container p-4">
                                <WaveformVisualizer 
                                    audioData={audioData}
                                    isRecording={isRecording}
                                    isPlaying={isPlaying}
                                />
                            </div>

                            {/* Controls */}
                            <div className="flex items-center justify-center gap-4 mt-6">
                                {!audioBlob ? (
                                    <Button
                                        size="lg"
                                        onClick={isRecording ? stopRecording : startRecording}
                                        className={`h-14 px-8 ${isRecording ? 'bg-destructive hover:bg-destructive/90' : 'bg-primary hover:bg-primary/90'} transition-all hover:scale-105 active:scale-95`}
                                        data-testid="record-btn"
                                    >
                                        {isRecording ? (
                                            <>
                                                <Square className="w-5 h-5 mr-2 recording-indicator" /> Stop Recording
                                            </>
                                        ) : (
                                            <>
                                                <Mic className="w-5 h-5 mr-2" /> Start Recording
                                            </>
                                        )}
                                    </Button>
                                ) : (
                                    <>
                                        <Button
                                            size="lg"
                                            variant="outline"
                                            onClick={handlePlayPause}
                                            className="h-14 w-14 p-0"
                                            data-testid="play-btn"
                                        >
                                            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                                        </Button>
                                        <Button
                                            size="lg"
                                            variant="outline"
                                            onClick={() => {
                                                setAudioBlob(null);
                                                setAudioUrl(null);
                                                setRecordingTime(0);
                                            }}
                                            className="h-14 px-6"
                                            data-testid="re-record-btn"
                                        >
                                            <RefreshCw className="w-4 h-4 mr-2" /> Re-record
                                        </Button>
                                    </>
                                )}
                            </div>

                            {/* Audio element for playback */}
                            {audioUrl && (
                                <audio 
                                    ref={audioRef} 
                                    src={audioUrl} 
                                    onEnded={() => setIsPlaying(false)}
                                />
                            )}
                        </div>

                        {/* Analysis Results */}
                        <AnimatePresence>
                            {project?.analysis && (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    className="bg-card border border-border/40 rounded-xl p-6"
                                >
                                    <h2 className="font-heading text-lg font-semibold mb-4 flex items-center gap-2">
                                        <Zap className="w-5 h-5 text-secondary" />
                                        AI Analysis
                                    </h2>
                                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                        <div className="bg-muted rounded-lg p-4">
                                            <span className="text-xs font-mono uppercase text-muted-foreground/70">BPM</span>
                                            <p className="font-heading text-2xl font-bold text-primary mt-1">
                                                {project.analysis.bpm}
                                            </p>
                                        </div>
                                        <div className="bg-muted rounded-lg p-4">
                                            <span className="text-xs font-mono uppercase text-muted-foreground/70">Flow Style</span>
                                            <p className="font-heading text-lg font-semibold mt-1 capitalize">
                                                {project.analysis.flow_style}
                                            </p>
                                        </div>
                                        <div className="bg-muted rounded-lg p-4">
                                            <span className="text-xs font-mono uppercase text-muted-foreground/70">Cadence</span>
                                            <p className="font-heading text-lg font-semibold mt-1 capitalize">
                                                {project.analysis.cadence}
                                            </p>
                                        </div>
                                        <div className="bg-muted rounded-lg p-4">
                                            <span className="text-xs font-mono uppercase text-muted-foreground/70">Mood</span>
                                            <p className="font-heading text-lg font-semibold mt-1 capitalize">
                                                {project.analysis.mood}
                                            </p>
                                        </div>
                                        <div className="bg-muted rounded-lg p-4">
                                            <span className="text-xs font-mono uppercase text-muted-foreground/70">Lyric Density</span>
                                            <p className="font-heading text-lg font-semibold mt-1 capitalize">
                                                {project.analysis.lyric_density}
                                            </p>
                                        </div>
                                        <div className="bg-muted rounded-lg p-4">
                                            <span className="text-xs font-mono uppercase text-muted-foreground/70">Suggested</span>
                                            <p className="text-sm mt-1 capitalize">
                                                {project.analysis.suggested_genres?.join(', ')}
                                            </p>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* Generated Beat */}
                        <AnimatePresence>
                            {project?.beat?.status === 'complete' && project?.beat?.audio_url && (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    className="bg-card border border-accent/40 rounded-xl p-6 glow-accent"
                                >
                                    <h2 className="font-heading text-lg font-semibold mb-4 flex items-center gap-2">
                                        <Check className="w-5 h-5 text-accent" />
                                        Your Beat is Ready!
                                    </h2>
                                    
                                    {project.beat.image_url && (
                                        <img 
                                            src={project.beat.image_url} 
                                            alt="Beat cover"
                                            className="w-full h-48 object-cover rounded-lg mb-4"
                                        />
                                    )}
                                    
                                    <div className="flex items-center gap-4 mb-4">
                                        <div className="flex-1">
                                            <p className="font-semibold">{project.beat.title || 'Generated Beat'}</p>
                                            <p className="text-sm text-muted-foreground">
                                                Duration: {Math.round(project.beat.duration || 0)}s
                                            </p>
                                            {typeof project.beat.repetition_score === 'number' && (
                                                <p className="text-sm text-muted-foreground">
                                                    Repetition score: {project.beat.repetition_score.toFixed(3)} (lower is better)
                                                </p>
                                            )}
                                            {typeof project.beat.render_attempts === 'number' && (
                                                <p className="text-xs text-muted-foreground/80">
                                                    Render attempts: {project.beat.render_attempts}
                                                </p>
                                            )}
                                            {project.beat.instrument_system_version && (
                                                <p className="text-xs text-muted-foreground/80">
                                                    Instruments: {project.beat.instrument_system_version}
                                                </p>
                                            )}
                                            {project.beat.instrument_pack && (
                                                <p className="text-xs text-muted-foreground/80 capitalize">
                                                    Pack: {project.beat.instrument_pack}
                                                </p>
                                            )}
                                            {project.beat.loop_source && (
                                                <p className="text-xs text-muted-foreground/80">
                                                    Loop: {project.beat.loop_source.split('\\').pop().split('/').pop()}
                                                </p>
                                            )}
                                            {project.beat.midi_sources && (
                                                <p className="text-xs text-muted-foreground/80">
                                                    MIDI: {[
                                                        project.beat.midi_sources.chords,
                                                        project.beat.midi_sources.melodies,
                                                        project.beat.midi_sources.basslines
                                                    ].filter(Boolean).map((p) => p.split('\\').pop().split('/').pop()).join(' | ') || 'none'}
                                                </p>
                                            )}
                                            {project.beat.asset_counts && (
                                                <p className="text-xs text-muted-foreground/80">
                                                    Assets: K{project.beat.asset_counts.kick_samples || 0} H{project.beat.asset_counts.hat_samples || 0} C{project.beat.asset_counts.clap_samples || 0} L{project.beat.asset_counts.loops || 0} M{project.beat.asset_counts.midi_files || 0}
                                                </p>
                                            )}
                                        </div>
                                    </div>

                                    <audio 
                                        controls 
                                        className="w-full mb-4"
                                        src={project.beat.audio_url}
                                    />

                                    {project.beat.mix_url && (
                                        <audio 
                                            controls 
                                            className="w-full mb-4"
                                            src={project.beat.mix_url}
                                        />
                                    )}

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <Button
                                            asChild
                                            className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
                                            data-testid="download-beat-btn"
                                        >
                                            <a href={project.beat.audio_url} download="accompaniment.wav">
                                                <Download className="w-4 h-4 mr-2" /> Download Beat
                                            </a>
                                        </Button>
                                        {project.beat.mix_url && (
                                            <Button
                                                asChild
                                                variant="outline"
                                                className="w-full"
                                                data-testid="download-mix-btn"
                                            >
                                                <a href={project.beat.mix_url} download="mix.wav">
                                                    <Download className="w-4 h-4 mr-2" /> Download Mix
                                                </a>
                                            </Button>
                                        )}
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Sidebar */}
                    <div className="space-y-6">
                        {/* Genre Selection */}
                        <div className="bg-card border border-border/40 rounded-xl p-6">
                            <h2 className="font-heading text-lg font-semibold mb-4">Beat Style</h2>
                            <Select 
                                value={project?.genre} 
                                onValueChange={handleGenreChange}
                            >
                                <SelectTrigger className="h-11 bg-input/50 border-transparent" data-testid="studio-genre-select">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {genres.map(g => (
                                        <SelectItem key={g.id} value={g.id}>
                                            <div className="flex flex-col">
                                                <span>{g.name}</span>
                                                <span className="text-xs text-muted-foreground">{g.description}</span>
                                            </div>
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        {/* Instrument Pack */}
                        <div className="bg-card border border-border/40 rounded-xl p-6">
                            <h2 className="font-heading text-lg font-semibold mb-4">Instrument Pack</h2>
                            <Select
                                value={project?.instrument_pack || 'auto'}
                                onValueChange={handleInstrumentPackChange}
                            >
                                <SelectTrigger className="h-11 bg-input/50 border-transparent" data-testid="studio-pack-select">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {INSTRUMENT_PACKS.map((p) => (
                                        <SelectItem key={p.id} value={p.id}>
                                            {p.name}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        {/* Volume Control */}
                        <div className="bg-card border border-border/40 rounded-xl p-6">
                            <h2 className="font-heading text-lg font-semibold mb-4 flex items-center gap-2">
                                <Volume2 className="w-4 h-4" /> Volume
                            </h2>
                            <Slider
                                value={[volume]}
                                onValueChange={handleVolumeChange}
                                max={100}
                                step={1}
                                className="w-full"
                                data-testid="volume-slider"
                            />
                            <span className="text-sm text-muted-foreground mt-2 block">{volume}%</span>
                        </div>

                        {/* Actions */}
                        <div className="bg-card border border-border/40 rounded-xl p-6 space-y-4">
                            <h2 className="font-heading text-lg font-semibold">Actions</h2>
                            
                            <Button
                                onClick={handleUploadAndAnalyze}
                                disabled={!audioBlob || isAnalyzing || project?.analysis}
                                className="w-full h-11 bg-secondary/10 text-secondary hover:bg-secondary/20 border border-secondary/20"
                                data-testid="analyze-btn"
                            >
                                {isAnalyzing ? (
                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                ) : project?.analysis ? (
                                    <Check className="w-4 h-4 mr-2" />
                                ) : (
                                    <Zap className="w-4 h-4 mr-2" />
                                )}
                                {isAnalyzing ? 'Analyzing...' : project?.analysis ? 'Analyzed' : 'Analyze Flow'}
                            </Button>

                            <Button
                                onClick={handleGenerateBeat}
                                disabled={!project?.analysis || isGenerating}
                                className="w-full h-11 bg-primary hover:bg-primary/90"
                                data-testid="generate-btn"
                            >
                                {isGenerating ? (
                                    <>
                                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                        Generating...
                                    </>
                                ) : project?.beat?.status === 'complete' ? (
                                    <>
                                        <RefreshCw className="w-4 h-4 mr-2" />
                                        Generate New Version
                                    </>
                                ) : (
                                    <>
                                        <Music className="w-4 h-4 mr-2" />
                                        Generate Beat
                                    </>
                                )}
                            </Button>

                            {isGenerating && (
                                <p className="text-xs text-muted-foreground text-center">
                                    <AlertCircle className="w-3 h-3 inline mr-1" />
                                    This may take 1-2 minutes...
                                </p>
                            )}
                        </div>

                        {/* Downloads Section */}
                        {(audioUrl || project?.beat?.status === 'complete') && (
                            <div className="bg-card border border-accent/40 rounded-xl p-6 space-y-4">
                                <h2 className="font-heading text-lg font-semibold flex items-center gap-2">
                                    <Download className="w-4 h-4 text-accent" /> Downloads
                                </h2>
                                
                                {/* Download Acapella */}
                                {audioUrl && (
                                    <Button
                                        asChild
                                        variant="outline"
                                        className="w-full h-11"
                                        data-testid="download-acapella-btn"
                                    >
                                        <a href={audioUrl} download="acapella.webm">
                                            <Mic className="w-4 h-4 mr-2" /> Download Acapella
                                        </a>
                                    </Button>
                                )}

                                {/* Download Beat */}
                                {project?.beat?.status === 'complete' && project?.beat?.audio_url && (
                                    <Button
                                        asChild
                                        className="w-full h-11 bg-accent hover:bg-accent/90 text-accent-foreground"
                                        data-testid="sidebar-download-beat-btn"
                                    >
                                        <a href={project.beat.audio_url} download="accompaniment.wav">
                                            <Music className="w-4 h-4 mr-2" /> Download Beat
                                        </a>
                                    </Button>
                                )}

                                {project?.beat?.status === 'complete' && project?.beat?.mix_url && (
                                    <Button
                                        asChild
                                        variant="outline"
                                        className="w-full h-11"
                                        data-testid="sidebar-download-mix-btn"
                                    >
                                        <a href={project.beat.mix_url} download="mix.wav">
                                            <Download className="w-4 h-4 mr-2" /> Download Mix
                                        </a>
                                    </Button>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
};
