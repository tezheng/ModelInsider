# Experimental Folder Structure Template

## Overview
This document defines the standard folder structure for organizing experimental and research work in the modelexport project. All experimental work should follow this structure to maintain consistency and organization.

## Standard Structure

```
experiments_sweeping/              # Root folder for all experimental work
├── README.md                     # Overview of current experiments and navigation guide
├── active/                       # Currently active experiments
│   ├── {experiment_name}/       # Individual experiment folders
│   │   ├── README.md           # Experiment description and status
│   │   ├── src/               # Source code
│   │   ├── tests/             # Test files
│   │   ├── results/           # Results and outputs
│   │   └── docs/              # Documentation specific to this experiment
│   │
│   └── temp_experiments/        # Quick tests and temporary work
│
├── iterations/                   # Iteration-based development work
│   ├── active/                  # Current iteration
│   │   └── iteration_{number}_{name}/
│   │       ├── plan.md         # Iteration plan
│   │       ├── notes.md        # Progress notes
│   │       └── results/        # Iteration results
│   │
│   ├── completed/               # Completed iterations
│   │   ├── iter_001_{name}/    # Completed iteration folders
│   │   ├── iter_002_{name}/
│   │   └── ...
│   │
│   └── templates/               # Templates for iterations
│       ├── iteration_template.md
│       └── iteration_checklist.md
│
├── benchmarks/                  # Performance benchmarking
│   ├── scripts/                # Benchmark scripts
│   ├── results/                # Benchmark results
│   │   └── {date}_{benchmark_name}/
│   └── analysis/               # Performance analysis
│
├── comparisons/                 # Comparison studies
│   ├── approaches/             # Different approach comparisons
│   ├── strategies/             # Strategy comparisons
│   ├── performance/            # Performance comparisons
│   └── results/                # Comparison results
│
├── prototypes/                  # Proof of concept implementations
│   ├── current/                # Active prototypes
│   │   └── {prototype_name}/
│   │       ├── README.md
│   │       ├── src/
│   │       └── demo/
│   │
│   └── archived/               # Old prototypes for reference
│
├── notebooks/                   # Jupyter notebooks for experiments
│   ├── analysis/               # Data analysis notebooks
│   ├── demos/                  # Demo notebooks
│   ├── explorations/          # Exploration and research
│   └── tutorials/             # Tutorial notebooks
│
├── scripts/                     # Experimental scripts
│   ├── investigation/          # Investigation scripts
│   ├── testing/               # Test scripts
│   ├── utilities/             # Utility scripts
│   └── automation/            # Automation scripts
│
└── archived/                    # Archived experiments
    ├── {year}/                 # Organized by year
    │   └── {experiment_name}/
    ├── legacy_implementations/  # Old implementations
    ├── deprecated_strategies/   # Deprecated approaches
    └── old_tests/              # Old test files
```

## Guidelines

### 1. Starting a New Experiment
When starting a new experiment:
1. Create a folder in `active/` with a descriptive name
2. Add a README.md with:
   - Experiment objective
   - Hypothesis
   - Methodology
   - Expected outcomes
   - Start date
   - Status (active/paused/completed)

### 2. Iteration Work
For iteration-based development:
1. Use sequential numbering: `iteration_XXX_description`
2. Always start with a plan.md
3. Document progress in notes.md
4. Move to `completed/` when done
5. Include summary of achievements and learnings

### 3. Benchmarking
For performance benchmarks:
1. Name results folders with date: `YYYY-MM-DD_benchmark_name`
2. Include configuration details
3. Store raw data and analysis separately
4. Document environment and hardware specs

### 4. Prototypes
For prototypes:
1. Keep in `current/` while actively developing
2. Include clear demo scripts
3. Document limitations and assumptions
4. Move to `archived/` when superseded

### 5. Archiving Policy
Move experiments to `archived/` when:
- The experiment is completed
- The approach is deprecated
- It hasn't been touched for 3+ months
- A better solution has been implemented

### 6. Naming Conventions
- Use lowercase with underscores for folders
- Use descriptive names that indicate purpose
- Include dates where relevant (YYYY-MM-DD format)
- Avoid special characters except underscore and hyphen

### 7. Documentation Requirements
Every experiment folder must have:
- README.md explaining the experiment
- Clear success criteria
- Dependencies listed
- How to run/reproduce the experiment

### 8. Results Storage
- Keep results with the experiment
- Use consistent formats (JSON, CSV, etc.)
- Include metadata (date, configuration, environment)
- Version large files separately (use Git LFS if needed)

## Example: Starting a New Experiment

```bash
# Create new experiment
experiments_sweeping/active/new_tagging_strategy/
├── README.md
├── src/
│   ├── __init__.py
│   └── new_tagger.py
├── tests/
│   └── test_new_tagger.py
├── results/
│   └── .gitkeep
└── docs/
    └── approach.md
```

## Migration Checklist
When migrating existing experimental work:
- [ ] Categorize as active/archived
- [ ] Add missing README files
- [ ] Organize by experiment type
- [ ] Update paths in documentation
- [ ] Remove duplicates
- [ ] Archive old iterations appropriately

## Maintenance
- Review `active/` monthly
- Archive completed work quarterly
- Clean up `temp_experiments/` weekly
- Update this template as needed

---

*Last Updated: 2025-08-01*
*Template Version: 1.0*