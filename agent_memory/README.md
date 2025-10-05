# Agent Memory Module

This module provides advanced memory and reasoning capabilities for AI medical research agents.

## Components

### 1. Live Reasoning (`live_reasoning.py`)
Pure Python multilingual medical/scientific reasoning agent with:
- Memory storage and retrieval
- PHI sanitization
- Configurable embedding dimensions
- TF-IDF weighting
- Persistent session management
- Adversarial token filtering

**Dependencies**: None (pure Python implementation)

### 2. Embed Memory (`embed_memory.py`)
Advanced centralized agent memory store with:
- PostgreSQL database backend
- Hybrid retrieval (semantic + keyword)
- PHI minimization and de-identification
- Optional encryption-at-rest
- Differential privacy support
- Association graph
- Audit logging

**Dependencies**: numpy, sqlalchemy, pydantic, PostgreSQL database

### 3. Imaging Insights (`imaging_insights.py`)
Radiology insight generation module featuring:
- Brain MRI volumetric analysis
- Z-score computation
- Risk stratification
- Quality control
- Configurable strategies via registry pattern

**Dependencies**: numpy, pandas, pydantic

## Running Demos

### Run All Demos
```bash
python agent_memory/run_all_demos.py
```

### Run Specific Demo
```bash
# Run only live_reasoning demo
python agent_memory/run_all_demos.py --only live_reasoning

# Run only imaging_insights demo (requires numpy, pandas)
python agent_memory/run_all_demos.py --only imaging_insights
```

### Skip Specific Demo
```bash
# Skip embed_memory (which requires PostgreSQL)
python agent_memory/run_all_demos.py --skip embed_memory
```

### Verbose Logging
```bash
python agent_memory/run_all_demos.py --verbose
```

## Individual Demo Execution

### Live Reasoning Demo
```bash
cd /path/to/AiMedRes
python -c "from agent_memory import live_reasoning; live_reasoning.demo()"
```

Or directly:
```bash
python agent_memory/live_reasoning.py
```

### Imaging Insights Demo
```bash
python agent_memory/imaging_insights.py
```

### Embed Memory Demo
Note: Requires PostgreSQL database configured
```bash
python agent_memory/embed_memory.py
```

## Integration with Training Pipeline

The agent_memory module demos can be integrated with the main training orchestrator:

```bash
# List all discovered training/demo scripts including agent_memory
python run_all_training.py --list

# Run specific training jobs
python run_all_training.py --only alzheimers
```

## Output

All demos produce:
- Console output with structured logging
- For live_reasoning: creates `memory_state_demo.json` (gitignored)
- Success/failure status with timing information

## Testing

The `run_all_demos.py` script provides:
- ✅ Success indicators for completed demos
- ⏭️ Skip indicators for demos with missing dependencies
- ❌ Failure indicators with error messages
- 📊 Summary statistics

Example output:
```
================================================================================
📊 Agent Memory Demo Summary
================================================================================
Total demos: 3
✅ Successful: 1
   - live_reasoning (0.02s)
⏭  Skipped: 2
   - embed_memory: Missing dependencies: No module named 'numpy'
   - imaging_insights: Missing dependencies: No module named 'numpy'
================================================================================
🎉 All runnable demos completed successfully!
```

## Notes

- The `live_reasoning` demo requires no external dependencies and will always run
- The `embed_memory` and `imaging_insights` demos require numpy/pandas and will be skipped if not available
- The `embed_memory` demo additionally requires a PostgreSQL database connection
- All demos include comprehensive error handling and graceful degradation
