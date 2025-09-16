# Atlas and ROI Mapping Assets

This directory contains anatomical atlases and region-of-interest (ROI) mapping assets for medical image analysis.

## Structure

```
atlas/
├── README.md                    # This file
├── atlas_config.yaml          # Atlas configuration
├── roi_mappings/               # ROI label mappings
│   ├── aal3_mapping.yaml      # AAL3 atlas ROI labels
│   ├── harvard_oxford_mapping.yaml  # Harvard-Oxford atlas labels
│   └── custom_mapping.yaml    # Custom ROI definitions
└── templates/                  # Reference templates (to be added)
    ├── MNI152_1mm.nii.gz      # MNI152 1mm template (link)
    ├── MNI152_2mm.nii.gz      # MNI152 2mm template (link)
    └── custom_template.nii.gz  # Custom template
```

## Atlas Sources

Atlas files are referenced via lightweight pointers to avoid large binary storage:

1. **MNI152 Templates**: Available from FSL or TemplateFlow
2. **AAL3 Atlas**: Automated Anatomical Labeling atlas version 3
3. **Harvard-Oxford Atlas**: Probabilistic atlas from FSL
4. **Custom Templates**: Project-specific atlases

## Usage

Atlas files can be downloaded automatically or referenced from standard neuroimaging packages like FSL, ANTs, or TemplateFlow.

For Git LFS tracked files, use:
```bash
git lfs pull
```

## Configuration

See `atlas_config.yaml` for atlas paths, download URLs, and metadata.