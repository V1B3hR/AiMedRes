#!/usr/bin/env python3
"""
Unified CLI structure for AiMedRes.

This provides the main entry point for the aimedres command-line tool with subcommands:
- aimedres train <options>
- aimedres serve <options>
- aimedres demo <options>
- aimedres validate <options>
"""

from __future__ import annotations

import argparse
import sys
import os
from typing import Optional, Sequence

# Add parent directory to path for imports when running as script
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = 'aimedres.cli'


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main entry point for the unified aimedres CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        prog='aimedres',
        description='AiMedRes - AI Medical Research Assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models
  aimedres train --epochs 50 --parallel --max-workers 4
  
  # Train specific model
  aimedres train --only alzheimers --epochs 30
  
  # Start API server
  aimedres serve --port 8000 --host 0.0.0.0
  
  # Run interactive mode
  aimedres interactive
  
  # Show version
  aimedres version
"""
    )
    
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version and exit'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train subcommand
    train_parser = subparsers.add_parser(
        'train',
        help='Train medical AI models',
        description='Run training pipelines for medical AI models'
    )
    train_parser.add_argument('--list', action='store_true', help='List available training jobs')
    train_parser.add_argument('--only', nargs='*', default=[], help='Train only these models')
    train_parser.add_argument('--exclude', nargs='*', default=[], help='Exclude these models')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--folds', type=int, help='Number of cross-validation folds')
    train_parser.add_argument('--batch', type=int, help='Batch size for training')
    train_parser.add_argument('--parallel', action='store_true', help='Run jobs in parallel')
    train_parser.add_argument('--max-workers', type=int, default=4, help='Max parallel workers')
    train_parser.add_argument('--dry-run', action='store_true', help='Show commands without executing')
    train_parser.add_argument('--config', type=str, help='Path to YAML config file')
    train_parser.add_argument('--no-auto-discover', action='store_true', help='Disable auto-discovery')
    train_parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    # Serve subcommand
    serve_parser = subparsers.add_parser(
        'serve',
        help='Start API server',
        description='Start the AiMedRes API server for remote training and inference'
    )
    serve_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    serve_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    serve_parser.add_argument('--ssl-cert', help='SSL certificate file for HTTPS')
    serve_parser.add_argument('--ssl-key', help='SSL private key file for HTTPS')
    
    # Interactive subcommand (legacy compatibility)
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Run interactive menu (legacy)',
        description='Run the legacy DuetMind interactive menu'
    )
    interactive_parser.add_argument('--config', type=str, help='Path to config file')
    interactive_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Handle version
    if args.version:
        try:
            from aimedres import __version__
            print(f"aimedres {__version__}")
        except ImportError:
            print("aimedres (version unknown)")
        return 0
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0
    
    # Dispatch to appropriate handler
    if args.command == 'train':
        # Run train module directly
        import subprocess
        train_path = os.path.join(os.path.dirname(__file__), 'train.py')
        # Convert args to argv format for train_cli
        train_argv = [sys.executable, train_path]
        if args.list:
            train_argv.append('--list')
        if args.only:
            train_argv.extend(['--only'] + args.only)
        if args.exclude:
            train_argv.extend(['--exclude'] + args.exclude)
        if args.epochs is not None:
            train_argv.extend(['--epochs', str(args.epochs)])
        if args.folds is not None:
            train_argv.extend(['--folds', str(args.folds)])
        if args.batch is not None:
            train_argv.extend(['--batch', str(args.batch)])
        if args.parallel:
            train_argv.append('--parallel')
            train_argv.extend(['--max-workers', str(args.max_workers)])
        if args.dry_run:
            train_argv.append('--dry-run')
        if args.config:
            train_argv.extend(['--config', args.config])
        if hasattr(args, 'no_auto_discover') and args.no_auto_discover:
            train_argv.append('--no-auto-discover')
        if hasattr(args, 'verbose') and args.verbose:
            train_argv.append('--verbose')
        
        result = subprocess.run(train_argv)
        return result.returncode
    
    elif args.command == 'serve':
        # Run serve module directly
        import subprocess
        serve_path = os.path.join(os.path.dirname(__file__), 'serve.py')
        # Convert args to argv format for serve_cli
        serve_argv = [
            sys.executable, serve_path,
            '--host', args.host,
            '--port', str(args.port),
        ]
        if args.debug:
            serve_argv.append('--debug')
        if args.ssl_cert:
            serve_argv.extend(['--ssl-cert', args.ssl_cert])
        if args.ssl_key:
            serve_argv.extend(['--ssl-key', args.ssl_key])
        
        result = subprocess.run(serve_argv)
        return result.returncode
    
    elif args.command == 'interactive':
        # Run the legacy interactive mode from __main__
        import subprocess
        main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '__main__.py')
        
        legacy_argv = [sys.executable, main_path, 'interactive']
        if args.config:
            legacy_argv.extend(['--config', args.config])
        if args.verbose:
            legacy_argv.append('--verbose')
        
        result = subprocess.run(legacy_argv)
        return result.returncode
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
