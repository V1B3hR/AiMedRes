# Quick Start: Training All Medical AI Models

**Version**: 1.0.0 | **Last Updated**: November 2025

This guide shows you the fastest way to train all 7 medical AI models.

## 🚀 Fastest Way to Get Started

### 1. Train All Models (Sequential)
```bash
./train_all_models.sh
```

### 2. Train All Models (Parallel - Faster!)
```bash
./train_all_models.sh --parallel --max-workers 4
```

### 3. Production Configuration
```bash
./train_all_models.sh --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128
```

## 📋 Available Models

When you run training, these 7 medical AI models will be trained:

1. **ALS** - Amyotrophic Lateral Sclerosis prediction
2. **Alzheimer's** - Cognitive decline detection
3. **Parkinson's** - Movement disorder prediction
4. **Brain MRI** - Brain tumor classification
5. **Cardiovascular** - Heart disease prediction
6. **Diabetes** - Diabetes risk assessment
7. **Specialized Agents** - Multi-agent medical system

## 🎯 Common Commands

### Preview Without Training (Dry Run)
```bash
./train_all_models.sh --dry-run
```

### Train Specific Models Only
```bash
./aimedres train --only als alzheimers parkinsons
```

### See All Options
```bash
./train_all_models.sh --help
./aimedres train --help
```

### List Available Models
```bash
./aimedres train --list
```

## 📊 What Happens When You Train?

1. **Execution**: Each model is trained with the specified parameters
2. **Logging**: Detailed logs are saved to `logs/`
3. **Results**: Model outputs are saved to `results/`
4. **Summary**: A JSON summary is saved to `summaries/`

## 📁 Output Structure

After training:
```
results/
├── als_comprehensive_results/
├── alzheimer_comprehensive_results/
├── parkinsons_comprehensive_results/
├── brain_mri_comprehensive_results/
├── cardiovascular_comprehensive_results/
├── diabetes_comprehensive_results/
└── specialized_agents_comprehensive_results/

logs/
├── orchestrator.log
└── [model_id]/run_[timestamp].log

summaries/
└── training_summary_[timestamp].json
```

## 🔧 Troubleshooting

### Not Enough Memory?
Use fewer parallel workers:
```bash
./train_all_models.sh --parallel --max-workers 2
```

### Want to Test First?
Use fewer epochs and dry-run:
```bash
./train_all_models.sh --dry-run --epochs 5 --folds 2
```

### Something Not Working?
Check the logs:
```bash
cat logs/orchestrator.log
```

## 📚 More Information

- **Complete Guide**: [RUN_ALL_MODELS_GUIDE.md](RUN_ALL_MODELS_GUIDE.md)
- **Implementation Details**: [TRAIN_ALL_MODELS_SUMMARY.md](TRAIN_ALL_MODELS_SUMMARY.md)
- **Demo Script**: Run `./run_all_models_demo.sh` for an interactive demo

## 💡 Tips

- Use `--parallel` for faster training on multi-core systems
- Start with `--dry-run` to preview commands
- Use `--epochs 5 --folds 2` for quick testing
- Use `--epochs 50 --folds 5` for production
- Add `--verbose` for more detailed output

## ✅ Success Indicators

You'll know training succeeded when you see:
```
🎉 All selected training pipelines completed successfully!
```

Check the summary file in `summaries/` for detailed results.
