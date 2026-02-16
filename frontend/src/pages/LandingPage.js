import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Mic, Music, Zap, Headphones, ArrowRight } from 'lucide-react';
import { Button } from '../components/ui/button';
import { useAuth } from '../context/AuthContext';

const features = [
    {
        icon: Mic,
        title: 'Record Your Flow',
        description: 'Spit your bars directly into the app with real-time waveform visualization.'
    },
    {
        icon: Zap,
        title: 'AI Analysis',
        description: 'Our AI analyzes your BPM, cadence, flow style, and mood to understand your vibe.'
    },
    {
        icon: Music,
        title: 'Custom Beats',
        description: 'Get instrumentals generated specifically to match your unique rap style.'
    },
    {
        icon: Headphones,
        title: 'Export & Share',
        description: 'Download your acapella and generated beat to mix and share your tracks.'
    }
];

const genres = ['Trap', 'Boom Bap', 'Drill', 'Lo-Fi', 'West Coast', 'Melodic'];

export const LandingPage = () => {
    const { isAuthenticated } = useAuth();
    const navigate = useNavigate();

    const handleGetStarted = () => {
        if (isAuthenticated) {
            navigate('/dashboard');
        } else {
            navigate('/login');
        }
    };

    return (
        <div className="min-h-screen bg-background">
            {/* Noise overlay */}
            <div className="noise-overlay" />
            
            {/* Background gradients */}
            <div className="fixed inset-0 bg-gradient-radial pointer-events-none" />
            <div className="fixed inset-0 bg-gradient-radial-secondary pointer-events-none" />

            {/* Header */}
            <header className="relative z-10 flex items-center justify-between px-6 md:px-12 py-6">
                <Link to="/" className="flex items-center gap-2">
                    <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                        <Mic className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-heading font-bold text-xl tracking-tight">FlowState</span>
                </Link>
                <nav className="flex items-center gap-4">
                    {isAuthenticated ? (
                        <Button 
                            onClick={() => navigate('/dashboard')}
                            data-testid="dashboard-btn"
                            className="bg-primary hover:bg-primary/90"
                        >
                            Dashboard
                        </Button>
                    ) : (
                        <>
                            <Button 
                                variant="ghost" 
                                onClick={() => navigate('/login')}
                                data-testid="login-btn"
                            >
                                Login
                            </Button>
                            <Button 
                                onClick={() => navigate('/register')}
                                data-testid="register-btn"
                                className="bg-primary hover:bg-primary/90"
                            >
                                Get Started
                            </Button>
                        </>
                    )}
                </nav>
            </header>

            {/* Hero Section */}
            <section className="relative z-10 px-6 md:px-12 pt-16 pb-24 md:pt-24 md:pb-32">
                <div className="max-w-7xl mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                        className="max-w-3xl"
                    >
                        <span className="inline-block px-4 py-1.5 mb-6 text-xs font-mono uppercase tracking-widest text-secondary bg-secondary/10 rounded-full border border-secondary/20">
                            AI-Powered Beat Forge
                        </span>
                        <h1 className="font-heading text-5xl md:text-7xl font-black tracking-tighter leading-none mb-6">
                            Rap First.<br />
                            <span className="text-primary">Beat Second.</span>
                        </h1>
                        <p className="text-lg md:text-xl text-muted-foreground leading-relaxed mb-8 max-w-2xl">
                            FlowState flips the script on beat making. Record your acapella, and let AI craft the perfect instrumental tailored to your unique flow, cadence, and style.
                        </p>
                        <div className="flex flex-wrap gap-4">
                            <Button 
                                size="lg" 
                                onClick={handleGetStarted}
                                data-testid="hero-get-started-btn"
                                className="bg-primary hover:bg-primary/90 h-12 px-8 text-base font-medium transition-all hover:scale-105 active:scale-95"
                            >
                                Start Creating <ArrowRight className="ml-2 w-4 h-4" />
                            </Button>
                            <Button 
                                size="lg" 
                                variant="outline"
                                className="h-12 px-8 text-base border-white/20 hover:bg-white/5"
                            >
                                Watch Demo
                            </Button>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Hero Image */}
            <section className="relative z-10 px-6 md:px-12 pb-24">
                <div className="max-w-7xl mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                        className="relative rounded-2xl overflow-hidden border border-white/10"
                    >
                        <img 
                            src="https://images.unsplash.com/photo-1741745978060-9add161ba2c2?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA1NTJ8MHwxfHNlYXJjaHwxfHxyYXBwZXIlMjByZWNvcmRpbmclMjBzdHVkaW8lMjBuZW9ufGVufDB8fHx8MTc3MTI3MzE1MXww&ixlib=rb-4.1.0&q=85"
                            alt="Studio recording"
                            className="w-full h-64 md:h-96 object-cover"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-background via-background/50 to-transparent" />
                        
                        {/* Floating waveform preview */}
                        <div className="absolute bottom-6 left-6 right-6 glass rounded-xl p-4">
                            <div className="flex items-center gap-4">
                                <div className="w-12 h-12 rounded-full bg-primary flex items-center justify-center glow-primary">
                                    <Mic className="w-6 h-6 text-white" />
                                </div>
                                <div className="flex-1">
                                    <div className="flex items-end gap-1 h-8">
                                        {[...Array(40)].map((_, i) => (
                                            <div 
                                                key={i}
                                                className="flex-1 bg-primary/60 rounded-t"
                                                style={{ height: `${Math.random() * 100}%` }}
                                            />
                                        ))}
                                    </div>
                                </div>
                                <div className="font-mono text-sm text-muted-foreground">
                                    00:32
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Genres */}
            <section className="relative z-10 px-6 md:px-12 pb-24">
                <div className="max-w-7xl mx-auto">
                    <div className="flex flex-wrap gap-3">
                        {genres.map((genre, i) => (
                            <motion.span
                                key={genre}
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: 0.1 * i }}
                                className="px-4 py-2 text-sm font-medium bg-muted rounded-full border border-border/40 hover:border-primary/50 transition-colors cursor-default"
                            >
                                {genre}
                            </motion.span>
                        ))}
                    </div>
                </div>
            </section>

            {/* Features */}
            <section className="relative z-10 px-6 md:px-12 pb-32">
                <div className="max-w-7xl mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="mb-12"
                    >
                        <span className="text-xs font-mono uppercase tracking-widest text-muted-foreground/70 mb-4 block">
                            How It Works
                        </span>
                        <h2 className="font-heading text-3xl md:text-5xl font-bold tracking-tight">
                            From Flow to Fire
                        </h2>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {features.map((feature, i) => (
                            <motion.div
                                key={feature.title}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: 0.1 * i }}
                                className="bg-card border border-border/40 rounded-xl p-6 hover:border-primary/50 transition-colors duration-300"
                            >
                                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                                    <feature.icon className="w-6 h-6 text-primary" />
                                </div>
                                <h3 className="font-heading text-lg font-semibold mb-2">{feature.title}</h3>
                                <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className="relative z-10 px-6 md:px-12 pb-32">
                <div className="max-w-7xl mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="bg-card border border-border/40 rounded-2xl p-8 md:p-12 relative overflow-hidden"
                    >
                        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent pointer-events-none" />
                        <div className="relative z-10 max-w-2xl">
                            <h2 className="font-heading text-3xl md:text-4xl font-bold tracking-tight mb-4">
                                Ready to find your sound?
                            </h2>
                            <p className="text-muted-foreground mb-6">
                                Join the new wave of artists who create beats that actually fit their flow. No more forcing your bars over generic instrumentals.
                            </p>
                            <Button 
                                size="lg" 
                                onClick={handleGetStarted}
                                data-testid="cta-get-started-btn"
                                className="bg-primary hover:bg-primary/90 h-12 px-8 transition-all hover:scale-105 active:scale-95"
                            >
                                Get Started Free <ArrowRight className="ml-2 w-4 h-4" />
                            </Button>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Footer */}
            <footer className="relative z-10 px-6 md:px-12 py-8 border-t border-border/40">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                            <Mic className="w-4 h-4 text-white" />
                        </div>
                        <span className="font-heading font-bold">FlowState</span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                        © 2024 FlowState. Built for artists, by artists.
                    </p>
                </div>
            </footer>
        </div>
    );
};
