import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Mic, Mail, Lock, User, ArrowRight, Loader2 } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { useAuth } from '../context/AuthContext';
import { toast } from 'sonner';

export const LoginPage = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            await login(email, password);
            toast.success('Welcome back!');
            navigate('/dashboard');
        } catch (error) {
            toast.error(error.response?.data?.detail || 'Login failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background flex">
            {/* Noise overlay */}
            <div className="noise-overlay" />
            
            {/* Left side - Form */}
            <div className="w-full lg:w-1/2 flex flex-col justify-center px-8 md:px-16 lg:px-24 py-12 relative z-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="max-w-md w-full mx-auto"
                >
                    <Link to="/" className="flex items-center gap-2 mb-12">
                        <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                            <Mic className="w-5 h-5 text-white" />
                        </div>
                        <span className="font-heading font-bold text-xl">FlowState</span>
                    </Link>

                    <h1 className="font-heading text-3xl md:text-4xl font-bold tracking-tight mb-2">
                        Welcome back
                    </h1>
                    <p className="text-muted-foreground mb-8">
                        Sign in to continue creating your sound.
                    </p>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="space-y-2">
                            <Label htmlFor="email" className="text-sm font-medium">Email</Label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                <Input
                                    id="email"
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@example.com"
                                    className="pl-10 h-11 bg-input/50 border-transparent focus:border-primary"
                                    data-testid="login-email-input"
                                    required
                                />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="password" className="text-sm font-medium">Password</Label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                <Input
                                    id="password"
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    className="pl-10 h-11 bg-input/50 border-transparent focus:border-primary"
                                    data-testid="login-password-input"
                                    required
                                />
                            </div>
                        </div>

                        <Button 
                            type="submit" 
                            className="w-full h-11 bg-primary hover:bg-primary/90 transition-all hover:scale-[1.02] active:scale-[0.98]"
                            disabled={loading}
                            data-testid="login-submit-btn"
                        >
                            {loading ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                                <>Sign In <ArrowRight className="ml-2 w-4 h-4" /></>
                            )}
                        </Button>
                    </form>

                    <p className="mt-8 text-sm text-muted-foreground text-center">
                        Don't have an account?{' '}
                        <Link to="/register" className="text-primary hover:underline" data-testid="register-link">
                            Create one
                        </Link>
                    </p>
                </motion.div>
            </div>

            {/* Right side - Image */}
            <div className="hidden lg:block lg:w-1/2 relative">
                <div className="absolute inset-0 bg-gradient-to-l from-transparent to-background z-10" />
                <img
                    src="https://images.unsplash.com/photo-1650147880756-32cff42ac2d7?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA3MDB8MHwxfHNlYXJjaHwxfHxhdWRpbyUyMG1peGluZyUyMGNvbnNvbGUlMjBrbm9icyUyMGNsb3NlJTIwdXB8ZW58MHx8fHwxNzcxMjczMTU3fDA&ixlib=rb-4.1.0&q=85"
                    alt="Mixing console"
                    className="w-full h-full object-cover"
                />
            </div>
        </div>
    );
};

export const RegisterPage = () => {
    const [email, setEmail] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const { register } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            await register(email, password, username);
            toast.success('Account created successfully!');
            navigate('/dashboard');
        } catch (error) {
            toast.error(error.response?.data?.detail || 'Registration failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background flex">
            {/* Noise overlay */}
            <div className="noise-overlay" />
            
            {/* Left side - Form */}
            <div className="w-full lg:w-1/2 flex flex-col justify-center px-8 md:px-16 lg:px-24 py-12 relative z-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="max-w-md w-full mx-auto"
                >
                    <Link to="/" className="flex items-center gap-2 mb-12">
                        <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                            <Mic className="w-5 h-5 text-white" />
                        </div>
                        <span className="font-heading font-bold text-xl">FlowState</span>
                    </Link>

                    <h1 className="font-heading text-3xl md:text-4xl font-bold tracking-tight mb-2">
                        Create your account
                    </h1>
                    <p className="text-muted-foreground mb-8">
                        Start creating beats that match your flow.
                    </p>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="space-y-2">
                            <Label htmlFor="username" className="text-sm font-medium">Username</Label>
                            <div className="relative">
                                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                <Input
                                    id="username"
                                    type="text"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    placeholder="yourname"
                                    className="pl-10 h-11 bg-input/50 border-transparent focus:border-primary"
                                    data-testid="register-username-input"
                                    required
                                />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="email" className="text-sm font-medium">Email</Label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                <Input
                                    id="email"
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@example.com"
                                    className="pl-10 h-11 bg-input/50 border-transparent focus:border-primary"
                                    data-testid="register-email-input"
                                    required
                                />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="password" className="text-sm font-medium">Password</Label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                <Input
                                    id="password"
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    className="pl-10 h-11 bg-input/50 border-transparent focus:border-primary"
                                    data-testid="register-password-input"
                                    required
                                    minLength={6}
                                />
                            </div>
                        </div>

                        <Button 
                            type="submit" 
                            className="w-full h-11 bg-primary hover:bg-primary/90 transition-all hover:scale-[1.02] active:scale-[0.98]"
                            disabled={loading}
                            data-testid="register-submit-btn"
                        >
                            {loading ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                                <>Create Account <ArrowRight className="ml-2 w-4 h-4" /></>
                            )}
                        </Button>
                    </form>

                    <p className="mt-8 text-sm text-muted-foreground text-center">
                        Already have an account?{' '}
                        <Link to="/login" className="text-primary hover:underline" data-testid="login-link">
                            Sign in
                        </Link>
                    </p>
                </motion.div>
            </div>

            {/* Right side - Image */}
            <div className="hidden lg:block lg:w-1/2 relative">
                <div className="absolute inset-0 bg-gradient-to-l from-transparent to-background z-10" />
                <img
                    src="https://images.unsplash.com/photo-1650147880756-32cff42ac2d7?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA3MDB8MHwxfHNlYXJjaHwxfHxhdWRpbyUyMG1peGluZyUyMGNvbnNvbGUlMjBrbm9icyUyMGNsb3NlJTIwdXB8ZW58MHx8fHwxNzcxMjczMTU3fDA&ixlib=rb-4.1.0&q=85"
                    alt="Mixing console"
                    className="w-full h-full object-cover"
                />
            </div>
        </div>
    );
};
