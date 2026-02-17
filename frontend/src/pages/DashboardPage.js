import React, { useState, useEffect, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
    Mic, Plus, Clock, Music, ChevronRight, LogOut, 
    Trash2, MoreVertical, Loader2 
} from 'lucide-react';
import { Button } from '../components/ui/button';
import { 
    DropdownMenu, 
    DropdownMenuContent, 
    DropdownMenuItem, 
    DropdownMenuTrigger 
} from '../components/ui/dropdown-menu';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from '../components/ui/dialog';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '../components/ui/select';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { useAuth } from '../context/AuthContext';
import { projectsApi, genresApi } from '../api/projects';
import { toast } from 'sonner';

const statusConfig = {
    draft: { label: 'Draft', class: 'status-draft' },
    uploaded: { label: 'Uploaded', class: 'status-draft' },
    analyzing: { label: 'Analyzing', class: 'status-analyzing' },
    analyzed: { label: 'Analyzed', class: 'status-analyzing' },
    generating: { label: 'Generating', class: 'status-generating' },
    complete: { label: 'Complete', class: 'status-complete' },
    error: { label: 'Error', class: 'status-error' }
};

export const DashboardPage = () => {
    const { user, token, logout } = useAuth();
    const navigate = useNavigate();
    const [projects, setProjects] = useState([]);
    const [genres, setGenres] = useState([]);
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [newProject, setNewProject] = useState({ name: '', genre: 'trap' });

    const loadData = useCallback(async () => {
        try {
            const [projectsData, genresData] = await Promise.all([
                projectsApi.getAll(token),
                genresApi.getAll()
            ]);
            setProjects(projectsData);
            setGenres(genresData);
        } catch (error) {
            toast.error('Failed to load projects');
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => {
        loadData();
    }, [loadData]);

    const handleCreateProject = async () => {
        if (!newProject.name.trim()) {
            toast.error('Please enter a project name');
            return;
        }
        setCreating(true);
        try {
            const project = await projectsApi.create(token, newProject);
            toast.success('Project created!');
            setDialogOpen(false);
            setNewProject({ name: '', genre: 'trap' });
            navigate(`/studio/${project.id}`);
        } catch (error) {
            toast.error('Failed to create project');
        } finally {
            setCreating(false);
        }
    };

    const handleDeleteProject = async (projectId) => {
        try {
            await projectsApi.delete(token, projectId);
            setProjects(projects.filter(p => p.id !== projectId));
            toast.success('Project deleted');
        } catch (error) {
            toast.error('Failed to delete project');
        }
    };

    const handleLogout = () => {
        logout();
        navigate('/');
    };

    const formatDate = (dateStr) => {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            year: 'numeric'
        });
    };

    return (
        <div className="min-h-screen bg-background">
            {/* Noise overlay */}
            <div className="noise-overlay" />
            
            {/* Background gradients */}
            <div className="fixed inset-0 bg-gradient-radial pointer-events-none" />

            {/* Header */}
            <header className="relative z-10 flex items-center justify-between px-6 md:px-12 py-6 border-b border-border/40">
                <Link to="/" className="flex items-center gap-2">
                    <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                        <Mic className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-heading font-bold text-xl tracking-tight">FlowState</span>
                </Link>
                <div className="flex items-center gap-4">
                    <span className="text-sm text-muted-foreground hidden md:block">
                        {user?.username}
                    </span>
                    <Button 
                        variant="ghost" 
                        size="icon"
                        onClick={handleLogout}
                        data-testid="logout-btn"
                    >
                        <LogOut className="w-4 h-4" />
                    </Button>
                </div>
            </header>

            {/* Main Content */}
            <main className="relative z-10 px-6 md:px-12 py-8 max-w-7xl mx-auto">
                {/* Header Row */}
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="font-heading text-3xl md:text-4xl font-bold tracking-tight">
                            Your Projects
                        </h1>
                        <p className="text-muted-foreground mt-1">
                            {projects.length} project{projects.length !== 1 ? 's' : ''}
                        </p>
                    </div>
                    
                    <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
                        <DialogTrigger asChild>
                            <Button 
                                className="bg-primary hover:bg-primary/90 h-11 px-6 transition-all hover:scale-105 active:scale-95"
                                data-testid="new-project-btn"
                            >
                                <Plus className="w-4 h-4 mr-2" /> New Project
                            </Button>
                        </DialogTrigger>
                        <DialogContent className="bg-card border-border/40">
                            <DialogHeader>
                                <DialogTitle className="font-heading text-xl">Create New Project</DialogTitle>
                            </DialogHeader>
                            <div className="space-y-4 mt-4">
                                <div className="space-y-2">
                                    <Label>Project Name</Label>
                                    <Input
                                        value={newProject.name}
                                        onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                                        placeholder="My Fire Track"
                                        className="h-11 bg-input/50 border-transparent focus:border-primary"
                                        data-testid="project-name-input"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Genre</Label>
                                    <Select 
                                        value={newProject.genre}
                                        onValueChange={(v) => setNewProject({ ...newProject, genre: v })}
                                    >
                                        <SelectTrigger className="h-11 bg-input/50 border-transparent" data-testid="genre-select">
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {genres.map(g => (
                                                <SelectItem key={g.id} value={g.id}>
                                                    {g.name}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <Button 
                                    onClick={handleCreateProject}
                                    className="w-full h-11 bg-primary hover:bg-primary/90"
                                    disabled={creating}
                                    data-testid="create-project-btn"
                                >
                                    {creating ? (
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                    ) : (
                                        'Create Project'
                                    )}
                                </Button>
                            </div>
                        </DialogContent>
                    </Dialog>
                </div>

                {/* Projects Grid */}
                {loading ? (
                    <div className="flex items-center justify-center h-64">
                        <Loader2 className="w-8 h-8 animate-spin text-primary" />
                    </div>
                ) : projects.length === 0 ? (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-center py-16"
                    >
                        <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
                            <Music className="w-8 h-8 text-muted-foreground" />
                        </div>
                        <h3 className="font-heading text-xl font-semibold mb-2">No projects yet</h3>
                        <p className="text-muted-foreground mb-6">
                            Create your first project to start making beats.
                        </p>
                        <Button 
                            onClick={() => setDialogOpen(true)}
                            className="bg-primary hover:bg-primary/90"
                            data-testid="empty-new-project-btn"
                        >
                            <Plus className="w-4 h-4 mr-2" /> Create Project
                        </Button>
                    </motion.div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {projects.map((project, i) => (
                            <motion.div
                                key={project.id}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.05 }}
                                className="bg-card border border-border/40 rounded-xl overflow-hidden hover:border-primary/50 transition-colors duration-300 group"
                            >
                                <Link 
                                    to={`/studio/${project.id}`}
                                    className="block p-6"
                                    data-testid={`project-card-${project.id}`}
                                >
                                    <div className="flex items-start justify-between mb-4">
                                        <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                                            <Mic className="w-6 h-6 text-primary" />
                                        </div>
                                        <DropdownMenu>
                                            <DropdownMenuTrigger asChild onClick={(e) => e.preventDefault()}>
                                                <Button variant="ghost" size="icon" className="h-8 w-8">
                                                    <MoreVertical className="w-4 h-4" />
                                                </Button>
                                            </DropdownMenuTrigger>
                                            <DropdownMenuContent align="end">
                                                <DropdownMenuItem 
                                                    className="text-destructive"
                                                    onClick={(e) => {
                                                        e.preventDefault();
                                                        handleDeleteProject(project.id);
                                                    }}
                                                >
                                                    <Trash2 className="w-4 h-4 mr-2" />
                                                    Delete
                                                </DropdownMenuItem>
                                            </DropdownMenuContent>
                                        </DropdownMenu>
                                    </div>
                                    <h3 className="font-heading text-lg font-semibold mb-1 group-hover:text-primary transition-colors">
                                        {project.name}
                                    </h3>
                                    <div className="flex items-center gap-3 text-sm text-muted-foreground mb-4">
                                        <span className="capitalize">{project.genre.replace('_', ' ')}</span>
                                        <span>•</span>
                                        <span className="flex items-center gap-1">
                                            <Clock className="w-3 h-3" />
                                            {formatDate(project.created_at)}
                                        </span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${statusConfig[project.status]?.class || 'status-draft'}`}>
                                            {statusConfig[project.status]?.label || project.status}
                                        </span>
                                        <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                                    </div>
                                </Link>
                            </motion.div>
                        ))}
                    </div>
                )}
            </main>
        </div>
    );
};
