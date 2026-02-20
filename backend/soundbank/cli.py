#!/usr/bin/env python3
"""
Sound Bank Builder - Main CLI Entry Point
==========================================
Unified CLI for all sound bank operations.

Usage:
    soundbank ingest ./samples -o ./output -c 808
    soundbank generate ./test_samples -n 10
    soundbank info ./output
    soundbank query ./output --intensity 0.15 --category loops
    soundbank export ./output 5 ./exported_sample.wav
"""

import argparse
import sys
from pathlib import Path


def cmd_ingest(args):
    """Handle ingest command."""
    from .ingest import IngestionEngine
    
    engine = IngestionEngine(
        output_dir=args.output,
        target_rms=args.target_rms,
        notch_attenuation_db=args.notch_db
    )
    
    processed, failed = engine.process_directory(
        input_dir=args.input_dir,
        category=args.category,
        apply_notch=not args.no_notch,
        normalize_mode=args.normalize,
        recursive=not args.no_recursive
    )
    
    return 0 if failed == 0 else 1


def cmd_generate(args):
    """Handle test sample generation command."""
    from .test_generator import TestSampleGenerator
    
    generator = TestSampleGenerator(output_dir=args.output)
    
    if args.categories:
        generator.create_category_sample_set(samples_per_category=args.count)
    else:
        generator.create_test_sample_set(
            count=args.count,
            duration_range=(args.duration * 0.5, args.duration * 1.5),
            include_markers=not args.no_markers
        )
    
    return 0


def cmd_info(args):
    """Handle info command."""
    from .provider import SoundBankProvider
    
    master_wav = Path(args.bank_dir) / "master_bank.wav"
    db_path = Path(args.bank_dir) / "bank.db"
    
    if not master_wav.exists() or not db_path.exists():
        print(f"Error: Sound bank not found in {args.bank_dir}")
        return 1
    
    provider = SoundBankProvider(str(master_wav), str(db_path))
    stats = provider.get_statistics()
    
    print("\n" + "="*50)
    print("SOUND BANK INFO")
    print("="*50)
    print(f"Master WAV: {stats['master_wav_path']}")
    print(f"Sample Rate: {stats['sample_rate']} Hz")
    print(f"Channels: {stats['channels']}")
    print(f"Total Duration: {stats['master_duration_seconds']:.2f} seconds")
    print(f"Total Assets: {stats['total_assets']}")
    print("\nAssets by Category:")
    for cat, count in stats.get('by_category', {}).items():
        print(f"  {cat}: {count}")
    print("\nIntensity Range:")
    intensity = stats.get('intensity_range', {})
    print(f"  Min: {intensity.get('min', 'N/A'):.4f}" if intensity.get('min') else "  Min: N/A")
    print(f"  Max: {intensity.get('max', 'N/A'):.4f}" if intensity.get('max') else "  Max: N/A")
    print(f"  Avg: {intensity.get('avg', 'N/A'):.4f}" if intensity.get('avg') else "  Avg: N/A")
    
    return 0


def cmd_query(args):
    """Handle query command."""
    from .provider import SoundBankProvider
    
    master_wav = Path(args.bank_dir) / "master_bank.wav"
    db_path = Path(args.bank_dir) / "bank.db"
    
    provider = SoundBankProvider(str(master_wav), str(db_path))
    
    if args.intensity is not None:
        results = provider.get_by_intensity(
            target_rms=args.intensity,
            tolerance=args.tolerance,
            category=args.category,
            return_all_matches=True
        )
        
        print(f"\nQuery: intensity={args.intensity} (±{args.tolerance})")
        if args.category:
            print(f"       category={args.category}")
        print(f"Found {len(results)} matches:\n")
        
        for asset, audio in results:
            duration_ms = len(audio) / provider.sample_rate * 1000
            print(f"  ID {asset.id}: {asset.original_filename}")
            print(f"    Category: {asset.category}")
            print(f"    Intensity: {asset.intensity_score:.4f}")
            print(f"    Duration: {duration_ms:.1f}ms")
            print(f"    Samples: {asset.start_sample} - {asset.end_sample}")
            print()
    else:
        assets = provider.list_assets(category=args.category)
        print(f"\nAll assets" + (f" in category '{args.category}'" if args.category else "") + ":\n")
        
        for asset in assets:
            print(f"  ID {asset.id}: {asset.original_filename} [{asset.category}] intensity={asset.intensity_score:.4f}")
    
    return 0


def cmd_export(args):
    """Handle export command."""
    from .provider import SoundBankProvider
    
    master_wav = Path(args.bank_dir) / "master_bank.wav"
    db_path = Path(args.bank_dir) / "bank.db"
    
    provider = SoundBankProvider(str(master_wav), str(db_path))
    
    output_path = provider.export_asset(
        asset_id=args.asset_id,
        output_path=args.output_file,
        format='WAV'
    )
    
    print(f"Exported asset {args.asset_id} to {output_path}")
    return 0


def cmd_verify(args):
    """Handle verify command."""
    from .test_generator import TestSampleGenerator
    
    generator = TestSampleGenerator()
    
    master_wav = Path(args.bank_dir) / "master_bank.wav"
    db_path = Path(args.bank_dir) / "bank.db"
    
    passed, errors = generator.verify_sample_accuracy(str(master_wav), str(db_path))
    
    return 0 if passed else 1


def cmd_spectral(args):
    """Handle spectral verification command."""
    from .provider import SoundBankProvider
    
    master_wav = Path(args.bank_dir) / "master_bank.wav"
    db_path = Path(args.bank_dir) / "bank.db"
    
    provider = SoundBankProvider(str(master_wav), str(db_path))
    
    if args.asset_id:
        # Verify single asset
        result = provider.verify_spectral_notch(asset_id=args.asset_id)
        asset = provider.get_asset_info(args.asset_id)
        
        print(f"\n{'='*50}")
        print(f"SPECTRAL ANALYSIS: Asset {args.asset_id}")
        print(f"{'='*50}")
        print(f"File: {asset.original_filename}")
        print(f"Category: {asset.category}")
        print(f"\nVocal Pocket Notch (1kHz-3kHz):")
        print(f"  Present: {'YES' if result.has_notch else 'NO'}")
        print(f"  Depth: {result.notch_depth_db:.1f} dB")
        print(f"  Confidence: {result.confidence*100:.0f}%")
        print(f"\nEnergy Levels:")
        print(f"  Surrounding bands: {result.full_band_energy:.1f} dB")
        print(f"  Notch band: {result.notch_band_energy:.1f} dB")
        
    else:
        # Verify all assets
        print(f"\n{'='*50}")
        print("SPECTRAL VERIFICATION: All Assets")
        print(f"{'='*50}")
        
        results = provider.verify_all_assets()
        
        passed = 0
        failed = 0
        
        for asset_id, result in results.items():
            asset = provider.get_asset_info(asset_id)
            status = "PASS" if result.has_notch else "FAIL"
            
            if result.has_notch:
                passed += 1
            else:
                failed += 1
            
            print(f"  [{status}] ID {asset_id}: {asset.original_filename} "
                  f"(notch: {result.notch_depth_db:.1f}dB, conf: {result.confidence*100:.0f}%)")
        
        print(f"\nSummary: {passed} passed, {failed} failed")
        print(f"Notch threshold: < {provider.NOTCH_THRESHOLD_DB} dB")
        
        return 0 if failed == 0 else 1
    
    return 0


def cmd_fetch(args):
    """Handle fetch command - get audio by intensity."""
    from .provider import SoundBankProvider
    import soundfile as sf
    
    master_wav = Path(args.bank_dir) / "master_bank.wav"
    db_path = Path(args.bank_dir) / "bank.db"
    
    provider = SoundBankProvider(str(master_wav), str(db_path))
    
    print(f"\nFetching audio with intensity={args.intensity} (normalized 0.0-1.0)")
    if args.category:
        print(f"Category filter: {args.category}")
    
    audio, asset, actual_intensity = provider.get_by_normalized_intensity_with_info(
        intensity=args.intensity,
        category=args.category
    )
    
    print(f"\nMatched Asset:")
    print(f"  ID: {asset.id}")
    print(f"  File: {asset.original_filename}")
    print(f"  Category: {asset.category}")
    print(f"  Requested intensity: {args.intensity:.2f}")
    print(f"  Actual intensity: {actual_intensity:.2f}")
    print(f"  Duration: {len(audio)/provider.sample_rate*1000:.1f}ms")
    
    if args.output:
        sf.write(args.output, audio, provider.sample_rate)
        print(f"\nSaved to: {args.output}")
    
    return 0


def cmd_transform(args):
    """Handle transformation command."""
    from .transformations import Transformations
    import soundfile as sf
    import numpy as np
    
    # Load audio
    audio, sr = sf.read(args.input_file)
    print(f"Loaded: {args.input_file} ({sr}Hz, {len(audio)} samples)")
    
    # Apply transformation
    if args.transform == 'notch':
        audio, hole = Transformations.apply_spectral_notch(
            audio, sr, attenuation_db=args.notch_db
        )
        print(f"Applied notch filter: {args.notch_db}dB @ 1-3kHz")
    
    elif args.transform == 'bitcrush':
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = Transformations.bit_crush(audio, bit_depth=args.bit_depth)
        print(f"Applied bit-crush: {args.bit_depth}-bit")
    
    elif args.transform == 'stretch':
        audio = Transformations.time_stretch(audio, stretch_factor=args.stretch_factor, sample_rate=sr)
        print(f"Applied time-stretch: {args.stretch_factor}x")
    
    # Save
    sf.write(args.output_file, audio, sr)
    print(f"Saved: {args.output_file}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='soundbank',
        description='Proprietary Sound Bank Builder - Create and manage addressable audio containers',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # INGEST command
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Process audio samples into Master WAV container'
    )
    ingest_parser.add_argument('input_dir', help='Input directory with audio files')
    ingest_parser.add_argument('-o', '--output', default='./output', help='Output directory')
    ingest_parser.add_argument('-c', '--category', default='loops',
                               choices=['808', 'snare', 'loops', 'atmospheres'])
    ingest_parser.add_argument('--normalize', choices=['rms', 'peak', 'none'], default='rms')
    ingest_parser.add_argument('--target-rms', type=float, default=0.1)
    ingest_parser.add_argument('--notch-db', type=float, default=-12.0)
    ingest_parser.add_argument('--no-notch', action='store_true')
    ingest_parser.add_argument('--no-recursive', action='store_true')
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # GENERATE command
    gen_parser = subparsers.add_parser(
        'generate',
        help='Generate test samples for validation'
    )
    gen_parser.add_argument('-o', '--output', default='./test_samples', help='Output directory')
    gen_parser.add_argument('-n', '--count', type=int, default=10, help='Number of samples')
    gen_parser.add_argument('--duration', type=float, default=0.5, help='Base duration (seconds)')
    gen_parser.add_argument('--categories', action='store_true', help='Organize by category')
    gen_parser.add_argument('--no-markers', action='store_true', help='Skip boundary markers')
    gen_parser.set_defaults(func=cmd_generate)
    
    # INFO command
    info_parser = subparsers.add_parser(
        'info',
        help='Display sound bank information'
    )
    info_parser.add_argument('bank_dir', help='Sound bank directory')
    info_parser.set_defaults(func=cmd_info)
    
    # QUERY command
    query_parser = subparsers.add_parser(
        'query',
        help='Query assets from sound bank'
    )
    query_parser.add_argument('bank_dir', help='Sound bank directory')
    query_parser.add_argument('-i', '--intensity', type=float, help='Target intensity (RMS)')
    query_parser.add_argument('-t', '--tolerance', type=float, default=0.1, help='Intensity tolerance')
    query_parser.add_argument('-c', '--category', help='Filter by category')
    query_parser.set_defaults(func=cmd_query)
    
    # EXPORT command
    export_parser = subparsers.add_parser(
        'export',
        help='Export single asset to file'
    )
    export_parser.add_argument('bank_dir', help='Sound bank directory')
    export_parser.add_argument('asset_id', type=int, help='Asset ID to export')
    export_parser.add_argument('output_file', help='Output file path')
    export_parser.set_defaults(func=cmd_export)
    
    # VERIFY command
    verify_parser = subparsers.add_parser(
        'verify',
        help='Verify sample accuracy of sound bank'
    )
    verify_parser.add_argument('bank_dir', help='Sound bank directory')
    verify_parser.set_defaults(func=cmd_verify)
    
    # SPECTRAL command
    spectral_parser = subparsers.add_parser(
        'spectral',
        help='Verify spectral notch (vocal pocket) is present'
    )
    spectral_parser.add_argument('bank_dir', help='Sound bank directory')
    spectral_parser.add_argument('-a', '--asset-id', type=int, help='Specific asset ID (or all if omitted)')
    spectral_parser.set_defaults(func=cmd_spectral)
    
    # FETCH command
    fetch_parser = subparsers.add_parser(
        'fetch',
        help='Fetch audio by normalized intensity (0.0-1.0)'
    )
    fetch_parser.add_argument('bank_dir', help='Sound bank directory')
    fetch_parser.add_argument('-i', '--intensity', type=float, required=True,
                              help='Intensity level (0.0=quietest, 1.0=loudest)')
    fetch_parser.add_argument('-c', '--category', help='Filter by category')
    fetch_parser.add_argument('-o', '--output', help='Save output to file')
    fetch_parser.set_defaults(func=cmd_fetch)
    
    # TRANSFORM command
    transform_parser = subparsers.add_parser(
        'transform',
        help='Apply transformation to audio file'
    )
    transform_parser.add_argument('input_file', help='Input audio file')
    transform_parser.add_argument('output_file', help='Output audio file')
    transform_parser.add_argument('-t', '--transform', required=True,
                                  choices=['notch', 'bitcrush', 'stretch'],
                                  help='Transformation to apply')
    transform_parser.add_argument('--notch-db', type=float, default=-12.0,
                                  help='Notch attenuation (dB)')
    transform_parser.add_argument('--bit-depth', type=int, default=8,
                                  help='Bit depth for bit-crush')
    transform_parser.add_argument('--stretch-factor', type=float, default=1.5,
                                  help='Time stretch factor')
    transform_parser.set_defaults(func=cmd_transform)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
