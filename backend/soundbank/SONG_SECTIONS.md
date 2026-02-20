"""
SONG SECTION TAGGING SYSTEM
===========================
Automatically identify which song sections each audio asset works in.

The classifier analyzes audio characteristics and tags it for:
- intro, verse, pre-chorus, chorus
- bridge, breakdown, build-up, drop, pre-drop
- fill, interlude, outro

This enables intelligent accompaniment generation that adapts to song structure.
"""

# =============================================================================
# SONG SECTION TAGS EXPLAINED
# =============================================================================

SONG_SECTIONS = {
    "intro": {
        "description": "Intro section - establishes vibe, builds anticipation",
        "characteristics": "Minimalist, calm, sparse, low-medium energy",
        "audio_profile": {
            "onset_density": "< 5 events/sec (sparse)",
            "rms_energy": "< 0.25 (quiet)",
            "attack": "> 50ms (soft entry)",
            "complexity": "Low to moderate"
        },
        "use_case": "Starting the song smoothly, setting tone",
        "example_instruments": ["ambient pad", "sparse drum loop", "soft bass line"]
    },
    
    "verse": {
        "description": "Verse section - main content, supports vocals",
        "characteristics": "Focused, clear, moderate energy, not cluttered",
        "audio_profile": {
            "onset_density": "2-10 events/sec (moderate)",
            "rms_energy": "< 0.35 (medium)",
            "attack": "10-150ms (medium attack)",
            "clarity": "Good frequency separation"
        },
        "use_case": "Supporting vocal performance without overpowering",
        "example_instruments": ["tight drums", "clean bass loop", "melodic keys"]
    },
    
    "pre-chorus": {
        "description": "Pre-chorus - builds tension toward main hook",
        "characteristics": "Rising energy, increasing movement, anticipatory",
        "audio_profile": {
            "spectral_flux": "> 0.25 (changing)",
            "onset_density": "> 5 events/sec (active)",
            "attack": "< 200ms (dynamic)",
            "trajectory": "Energy increases"
        },
        "use_case": "Creating anticipation before chorus drop",
        "example_instruments": ["building drum rolls", "rising synth", "ascending bass"]
    },
    
    "chorus": {
        "description": "Chorus - main hook, maximum impact, memorable",
        "characteristics": "HIGH energy, punchy, full arrangement, impactful",
        "audio_profile": {
            "onset_density": "> 8 events/sec (dense)",
            "rms_energy": "> 0.25 (loud)",
            "attack": "< 50ms (punchy)",
            "presence": "Full frequency spectrum"
        },
        "use_case": "Catchy, memorable, stands out in the song",
        "example_instruments": ["energetic drums", "full bass", "bright synths", "stabs"]
    },
    
    "bridge": {
        "description": "Bridge - contrasting element, breaks pattern",
        "characteristics": "Different from main pattern, unexpected texture",
        "audio_profile": {
            "pattern_change": "Contrasts with verse/chorus",
            "either": {
                "option1": "Much sparser (< 2 events/sec, minimal)",
                "option2": "Much denser (> 15 events/sec, complex)"
            }
        },
        "use_case": "Fresh perspective, variation, prevents monotony",
        "example_instruments": ["stripped-down drums", "unexpected synth", "textural pad"]
    },
    
    "breakdown": {
        "description": "Breakdown - reduced elements, tension drop",
        "characteristics": "Minimalist, sparse, releases tension, allows breathing room",
        "audio_profile": {
            "onset_density": "< 3 events/sec (very sparse)",
            "complexity": "< 0.15 spectral flux (simple)",
            "rms_energy": "< 0.2 (quiet)",
            "trajectory": "Simplifying"
        },
        "use_case": "Dropping energy to build anticipation for next peak",
        "example_instruments": ["minimal drums", "solo bass", "atmospheric pad"]
    },
    
    "build-up": {
        "description": "Build-up - increasing energy, rising momentum",
        "characteristics": "Progressive energy increase, mounting intensity",
        "audio_profile": {
            "spectral_flux": "0.2-0.6 (moderate-to-high change)",
            "onset_density": "5-15 events/sec (building)",
            "trajectory": "Progressive rise"
        },
        "use_case": "Leading listener to climax/drop with anticipation",
        "example_instruments": ["drum rolls", "rising synth", "bass roll"]
    },
    
    "drop": {
        "description": "Drop/Climax - released tension, maximum peak",
        "characteristics": "Explosive energy, sudden peak, climactic",
        "audio_profile": {
            "onset_density": "> 12 events/sec (very dense)",
            "rms_energy": "> 0.35 (very loud)",
            "attack": "< 10ms (super punchy)",
            "presence": "Ultra-present"
        },
        "use_case": "Most memorable, highest energy moment in song",
        "example_instruments": ["heavy drums", "powerful bass", "bright leads"]
    },
    
    "pre-drop": {
        "description": "Pre-drop - tension before release",
        "characteristics": "Teaser, anticipatory, high change with held-back density",
        "audio_profile": {
            "spectral_flux": "> 0.3 (high flux)",
            "onset_density": "5-10 events/sec (medium, held back)",
            "strategy": "Maximum movement with restrained events"
        },
        "use_case": "Final tension before the ultimate release (drop)",
        "example_instruments": ["anxious drum fills", "wobbling synth", "tension bass"]
    },
    
    "fill": {
        "description": "Fill/Transition - short connector, smooth transition",
        "characteristics": "Short duration, fills gaps, connects sections",
        "audio_profile": {
            "duration": "< 2 seconds (brief)",
            "onset_density": "> 5 events/sec (active despite brevity)",
            "purpose": "Bridging sections"
        },
        "use_case": "Smoothing transitions between major sections",
        "example_instruments": ["quick drum fill", "cymbal swell", "quick synth stab"]
    },
    
    "interlude": {
        "description": "Interlude - instrumental break, different texture",
        "characteristics": "Atmospheric, sustained elements, textural",
        "audio_profile": {
            "onset_density": "< 8 events/sec (moderate)",
            "spectral_variation": "High variance (textural depth)",
            "sustained": "Long tails, not percussive"
        },
        "use_case": "Taking center stage as instrumental moment",
        "example_instruments": ["ambient pad", "atmospheric synth", "sustained strings"]
    },
    
    "outro": {
        "description": "Outro - wind-down, conclusion",
        "characteristics": "Decreasing energy, winding down, resolution",
        "audio_profile": {
            "tail_energy": "< 15% of peak (fades away)",
            "onset_density": "< 7 events/sec (reducing)",
            "trajectory": "Decreasing energy"
        },
        "use_case": "Graceful exit from the song",
        "example_instruments": ["fading drums", "decaying pad", "reverb-heavy tail"]
    }
}

# =============================================================================
# AUTO-DETECTION ALGORITHM
# =============================================================================

"""
The classifier uses audio metrics to infer section suitability:

1. ONSET DENSITY (Events per second)
   - Sparse (<3): Good for intro, breakdown, outro
   - Moderate (5-10): Good for verse, pre-chorus
   - Dense (>12): Good for chorus, drop
   - Rising pattern: Good for build-up

2. SPECTRAL FLUX (Rate of frequency change)
   - Low (<0.15): Intro, breakdown, outro (stable)
   - Moderate (0.2-0.6): Build-up, pre-drop (movement)
   - High (>0.3): Pre-drop, interlude (change)

3. RMS ENERGY (Overall loudness)
   - Quiet (<0.2): Intro, breakdown
   - Medium (0.25-0.35): Verse, pre-chorus
   - Loud (>0.35): Chorus, drop

4. ATTACK TIME (How quickly sound reaches peak)
   - Slow (>50ms): Intro, breakdown (soft)
   - Medium (10-150ms): Verse, pre-chorus
   - Fast (<50ms): Chorus
   - Super-fast (<10ms): Drop (punchy)

5. FREQUENCY VARIATION (Spectral variety)
   - Low: Focused, single instruments
   - High: Textured, complex arrangements
   - Good for different sections based on context
"""

# =============================================================================
# PRACTICAL QUERIES
# =============================================================================

"""
BUILD A SONG STRUCTURE AUTOMATICALLY:

from soundbank.provider import SoundBankProvider

provider = SoundBankProvider('output/master_bank.wav', 'output/bank.db')

# Create a complete arrangement from intro to outro
song_structure = {
    'intro': provider.get_by_tag('intro', limit=5),
    'verse_1': provider.get_by_tag('verse', intensity_target=0.4, limit=5),
    'pre_chorus': provider.get_by_tag('pre-chorus', intensity_target=0.6, limit=3),
    'chorus': provider.get_by_tag('chorus', intensity_target=0.9, limit=5),
    'verse_2': provider.get_by_tag('verse', intensity_target=0.5, limit=5),
    'pre_chorus_2': provider.get_by_tag('pre-chorus', intensity_target=0.65, limit=3),
    'chorus_2': provider.get_by_tag('chorus', intensity_target=0.95, limit=5),
    'bridge': provider.get_by_tag('bridge', intensity_target=0.3, limit=3),
    'build_up': provider.get_by_tag('build-up', intensity_target=0.7, limit=3),
    'drop': provider.get_by_tag('drop', intensity_target=1.0, limit=3),
    'outro': provider.get_by_tag('outro', intensity_target=0.2, limit=3),
}

# Each value contains up to 5 suitable options for that section


QUERY BY MULTIPLE SECTION TAGS:

# Get samples that work in both verses AND interludes (versatile)
versatile = provider.get_by_tags(
    tags=['verse', 'interlude'],
    match_all=False,
    limit=20
)

# Get intro + build-up combo (progressive opening)
progressive_opening = provider.get_by_tags(
    tags=['intro', 'build-up'],
    match_all=False,
    limit=10
)


DYNAMIC SONG GENERATION:

def generate_song_section(section_type, optional_intensity=None):
    '''
    Generate accompaniment for specific song section.
    
    section_type: 'intro', 'verse', 'chorus', 'bridge', 'drop', etc.
    optional_intensity: 0.0-1.0, or auto-detect from vocal
    '''
    
    try:
        if optional_intensity:
            results = provider.get_by_tags(
                [section_type],
                intensity_target=optional_intensity,
                limit=5
            )
        else:
            results = provider.get_by_tag(section_type, limit=5)
        
        # Pick best match
        if results:
            if isinstance(results, tuple):
                asset, audio = results
            else:
                asset, audio = results[0]
            
            return audio, asset
    except:
        pass
    
    # Fallback to generic assets
    return None, None


USE IN ACS GENERATOR:

def generate_accompaniment(vocal_audio, song_section='verse'):
    '''
    Generate accompaniment tailored to song section.
    '''
    
    # Analyze vocal
    vocal_intensity = analyze_vocal_density(vocal_audio)
    
    # Get section-matched accompaniment
    loop_audio, loop_asset = generate_song_section(
        section_type=song_section,
        optional_intensity=vocal_intensity
    )
    
    if loop_audio is not None:
        # Use section-matched loop
        accompaniment = mix_with_vocal(vocal_audio, loop_audio)
        tags = provider.get_asset_tags(loop_asset.id)
        print(f"Using {song_section} loop: {loop_asset.original_filename}")
        print(f"Tags: {tags}")
        
        return accompaniment
    else:
        # Fallback to generic accompaniment
        return generic_accompaniment(vocal_audio)
"""

# =============================================================================
# SONG STRUCTURE EXAMPLES
# =============================================================================

"""
TYPICAL POP SONG (3:30)
─────────────────────

0:00-0:20  INTRO        [intro tag] - Establishes vibe
0:20-0:50  VERSE 1      [verse tag] - Introduces theme
0:50-1:10  PRE-CHORUS   [pre-chorus tag] - Builds anticipation
1:10-1:30  CHORUS       [chorus tag] - Main hook (high energy)
1:30-2:00  VERSE 2      [verse tag] - Develops theme
2:00-2:20  PRE-CHORUS   [pre-chorus tag] - Builds again
2:20-2:40  CHORUS       [chorus tag] - Repeat of hook
2:40-3:00  BRIDGE       [bridge tag] - Contrast/variation
3:00-3:20  FINAL CHORUS [chorus tag] - Last hook
3:20-3:30  OUTRO        [outro tag] - Wind down


EDGY EDM DROP (3:00)
────────────────────

0:00-0:20  INTRO        [intro tag] - Soft start
0:20-1:00  BUILD-UP     [build-up tag] - Rising tension
1:00-1:30  DROP         [drop tag] - CLIMAX! Maximum energy
1:30-2:00  BUILD-UP     [build-up tag] - Rebuild
2:00-2:30  DROP         [drop tag] - Second climax
2:30-3:00  OUTRO        [outro tag] - Fadeout


HIP-HOP CYPHER (4:00)
─────────────────────

0:00-0:30  INTRO        [intro tag] - Beat introduction
0:30-1:30  VERSE 1      [verse tag] - First rapper
1:30-2:00  INTERLUDE    [interlude tag] - Beat showcase
2:00-3:00  VERSE 2      [verse tag] - Second rapper
3:00-3:30  BREAKDOWN    [breakdown tag] - Stopping point
3:30-4:00  OUTRO        [outro tag] - Loop fades


EXPERIMENTAL/PROGRESSIVE (5:00)
────────────────────────────────

0:00-1:00  INTRO        [intro tag] - Ambient beginning
1:00-2:00  BREAKDOWN    [breakdown tag] - Sparse, minimal
2:00-3:00  BUILD-UP     [build-up tag] - Gradual intensification
3:00-4:00  DROP         [drop tag] - Payoff
4:00-5:00  OUTRO        [outro tag] - Resolution/fadeout
"""

# =============================================================================
# TAG COMBINATION LOGIC
# =============================================================================

"""
Most tracks get MULTIPLE section tags based on characteristics:

Example 1: A drum fill
  Tags: ['fill', 'percussive', 'fast-attack']
  Fits: Transitions between any sections
  
Example 2: An atmospheric pad
  Tags: ['intro', 'breakdown', 'interlude', 'outro', 'ambient']
  Fits: Quiet sections, atmospheric moments
  
Example 3: An energetic bass loop
  Tags: ['chorus', 'drop', 'high-energy', 'bass']
  Fits: Main moments, climaxes
  
Example 4: A versatile drum pattern
  Tags: ['verse', 'pre-chorus', 'build-up']
  Fits: Multiple sections, progressive energy rise


FILTERING LOGIC:

If composing a VERSE → Want: [verse | melodic-friendly] tags
If composing a CHORUS → Want: [chorus | punchy | high-energy]
If composing a BRIDGE → Want: [bridge | contrasting | organic]
If composer OUTRO → Want: [outro | fadeout | decreasing-energy]
"""

print(__doc__)
