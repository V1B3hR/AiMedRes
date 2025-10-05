#!/usr/bin/env python3
"""
Agent Memory Demo Runner
========================

This script runs all demo/training functions in the agent_memory module.
It systematically exercises each component's demo functionality:
- live_reasoning: Pure Python reasoning agent with memory
- embed_memory: Advanced memory store with database (requires PostgreSQL)
- imaging_insights: Radiology insight generation module

Usage:
    python agent_memory/run_all_demos.py
    python agent_memory/run_all_demos.py --only live_reasoning
    python agent_memory/run_all_demos.py --skip embed_memory
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AgentMemoryRunner")


class DemoResult:
    """Result of running a single demo"""
    def __init__(self, name: str):
        self.name = name
        self.status = "PENDING"
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.duration_sec: Optional[float] = None
        self.error: Optional[str] = None
        
    def __repr__(self):
        return f"DemoResult(name={self.name}, status={self.status}, duration={self.duration_sec})"


def run_live_reasoning_demo() -> DemoResult:
    """Run the live_reasoning demo"""
    result = DemoResult("live_reasoning")
    result.start_time = datetime.utcnow().isoformat()
    start = time.time()
    
    try:
        logger.info("=" * 80)
        logger.info("Running live_reasoning demo...")
        logger.info("=" * 80)
        
        # Import and run demo - try multiple import paths
        demo = None
        try:
            # Try new location first
            from src.aimedres.agent_memory.live_reasoning import demo
        except ImportError:
            try:
                # Try old location
                import sys
                from pathlib import Path
                agent_memory_path = Path(__file__).parent
                sys.path.insert(0, str(agent_memory_path))
                import live_reasoning
                demo = live_reasoning.demo
            except ImportError as e:
                raise ImportError(f"Could not import live_reasoning from any location: {e}")
        
        demo()
        
        result.status = "SUCCESS"
        logger.info("✅ live_reasoning demo completed successfully")
    except Exception as e:
        result.status = "FAILED"
        result.error = str(e)
        logger.error(f"❌ live_reasoning demo failed: {e}")
        logger.exception(e)
    finally:
        result.end_time = datetime.utcnow().isoformat()
        result.duration_sec = round(time.time() - start, 2)
    
    return result


def run_embed_memory_demo() -> DemoResult:
    """Run the embed_memory demo"""
    result = DemoResult("embed_memory")
    result.start_time = datetime.utcnow().isoformat()
    start = time.time()
    
    try:
        logger.info("=" * 80)
        logger.info("Running embed_memory demo...")
        logger.info("=" * 80)
        
        # Check for dependencies
        try:
            import numpy as np
            from sqlalchemy import create_engine
        except ImportError as e:
            logger.warning(f"⚠️  Skipping embed_memory: missing dependencies - {e}")
            result.status = "SKIPPED"
            result.error = f"Missing dependencies: {e}"
            return result
        
        # Check for PostgreSQL
        # Note: This demo requires PostgreSQL which may not be available
        logger.info("⚠️  embed_memory requires PostgreSQL database")
        logger.info("⚠️  Skipping embed_memory demo (requires external database)")
        result.status = "SKIPPED"
        result.error = "Requires PostgreSQL database"
        
    except Exception as e:
        result.status = "FAILED"
        result.error = str(e)
        logger.error(f"❌ embed_memory demo failed: {e}")
        logger.exception(e)
    finally:
        result.end_time = datetime.utcnow().isoformat()
        result.duration_sec = round(time.time() - start, 2)
    
    return result


def run_imaging_insights_demo() -> DemoResult:
    """Run the imaging_insights demo"""
    result = DemoResult("imaging_insights")
    result.start_time = datetime.utcnow().isoformat()
    start = time.time()
    
    try:
        logger.info("=" * 80)
        logger.info("Running imaging_insights demo...")
        logger.info("=" * 80)
        
        # Check for dependencies
        try:
            import numpy as np
            import pandas as pd
        except ImportError as e:
            logger.warning(f"⚠️  Skipping imaging_insights: missing dependencies - {e}")
            result.status = "SKIPPED"
            result.error = f"Missing dependencies: {e}"
            return result
        
        # Import and run example - try multiple import paths
        _example_usage = None
        try:
            # Try new location first
            from src.aimedres.agent_memory.imaging_insights import _example_usage
        except ImportError:
            try:
                # Try old location
                import sys
                from pathlib import Path
                agent_memory_path = Path(__file__).parent
                sys.path.insert(0, str(agent_memory_path))
                import imaging_insights
                _example_usage = imaging_insights._example_usage
            except ImportError as e:
                raise ImportError(f"Could not import imaging_insights from any location: {e}")
        
        _example_usage()
        
        result.status = "SUCCESS"
        logger.info("✅ imaging_insights demo completed successfully")
    except Exception as e:
        result.status = "FAILED"
        result.error = str(e)
        logger.error(f"❌ imaging_insights demo failed: {e}")
        logger.exception(e)
    finally:
        result.end_time = datetime.utcnow().isoformat()
        result.duration_sec = round(time.time() - start, 2)
    
    return result


def print_summary(results: List[DemoResult]):
    """Print summary of all demo runs"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 Agent Memory Demo Summary")
    logger.info("=" * 80)
    
    success = [r for r in results if r.status == "SUCCESS"]
    failed = [r for r in results if r.status == "FAILED"]
    skipped = [r for r in results if r.status == "SKIPPED"]
    
    logger.info(f"Total demos: {len(results)}")
    logger.info(f"✅ Successful: {len(success)}")
    if success:
        for r in success:
            logger.info(f"   - {r.name} ({r.duration_sec}s)")
    
    if skipped:
        logger.info(f"⏭  Skipped: {len(skipped)}")
        for r in skipped:
            logger.info(f"   - {r.name}: {r.error}")
    
    if failed:
        logger.info(f"❌ Failed: {len(failed)}")
        for r in failed:
            logger.info(f"   - {r.name}: {r.error}")
    
    logger.info("=" * 80)
    
    if failed:
        logger.warning("⚠️  Some demos failed. Check logs for details.")
        return 1
    elif not success and skipped:
        logger.warning("⚠️  All demos were skipped (missing dependencies).")
        return 0
    else:
        logger.info("🎉 All runnable demos completed successfully!")
        return 0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run all demo functions in agent_memory module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all demos:
    python agent_memory/run_all_demos.py
    
  Run specific demo:
    python agent_memory/run_all_demos.py --only live_reasoning
    
  Skip specific demo:
    python agent_memory/run_all_demos.py --skip embed_memory
        """
    )
    parser.add_argument(
        '--only',
        nargs='+',
        choices=['live_reasoning', 'embed_memory', 'imaging_insights'],
        help='Run only specified demos'
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        choices=['live_reasoning', 'embed_memory', 'imaging_insights'],
        help='Skip specified demos'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("🚀 Agent Memory Demo Runner")
    logger.info("=" * 80)
    logger.info(f"⏰ Started at (UTC): {datetime.utcnow().isoformat()}")
    
    # Determine which demos to run
    all_demos = ['live_reasoning', 'embed_memory', 'imaging_insights']
    
    if args.only:
        demos_to_run = args.only
    else:
        demos_to_run = all_demos
    
    if args.skip:
        demos_to_run = [d for d in demos_to_run if d not in args.skip]
    
    logger.info(f"Demos to run: {', '.join(demos_to_run)}")
    
    # Run demos
    results = []
    
    if 'live_reasoning' in demos_to_run:
        results.append(run_live_reasoning_demo())
    
    if 'embed_memory' in demos_to_run:
        results.append(run_embed_memory_demo())
    
    if 'imaging_insights' in demos_to_run:
        results.append(run_imaging_insights_demo())
    
    # Print summary
    exit_code = print_summary(results)
    
    logger.info(f"⏰ Finished at (UTC): {datetime.utcnow().isoformat()}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
