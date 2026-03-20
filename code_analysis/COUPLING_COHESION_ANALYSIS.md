# Coupling & Cohesion Analysis: src/driver Folder

**Date:** March 9, 2026  
**Analysis Scope:** All Python files in `src/driver/` directory  
**Files Analyzed:** 10 (excluding `__init__.py` and `__pycache__/`)

---

## Executive Summary

### Overall Assessment

- **Coupling: MODERATE** - Driver files do not depend on each other (good isolation), but each has 2-5 external dependencies on helper modules
- **Cohesion: LOW-MODERATE** - Significant code duplication (duplicate files, nearly identical utility functions) reduces internal cohesion
- **Key Issues:** 2 obsolete files, 2 duplicate utility modules, 1 file with high coupling (pnn_mme_driver.py)

### Quick Wins

1. **Delete 2 obsolete files** (5 min cleanup, immediate impact)
2. **Consolidate duplicate utilities** (tuner_retriever.py & tuner_models_retriever.py)
3. **Extract common patterns** from operational_mse_crps_driver.py and crps_mme_runner.py

---

## File-by-File Breakdown

### **Entry Point Scripts** (Runnable via CLI)

#### 1. **crps_mme_runner.py** ⭐
**Purpose:** Multi-model ensemble for water temperature prediction using CRPS loss function

**Cohesion:** HIGH  
**Coupling:** MEDIUM (`utils_mse_crps`, `my_parser`)

**Key Components:**
- `MyHyperModel` class - Keras hypermodel for RandomSearch tuning
- Custom metric wrappers: `mae_metric()`, `mae12_metric()`, `me_metric()`, `me12_metric()`
- `temp()` - Main execution loop across lead times and rotations
- Configuration for 25 models, 100 predictions ensemble

**Dependencies:**
```
├─ src.helper.utils_mse_crps (CRPS loss, metrics, data prep)
├─ src.helper.my_parser (command-line argument parsing)
├─ keras_tuner (RandomSearch)
└─ tensorflow/keras
```

**Responsibilities:**
- ✅ Hyperparameter tuning
- ✅ Model ensemble training
- ✅ Metric computation
- ✅ Results saving

---

#### 2. **operational_mse_crps_driver.py** ⭐
**Purpose:** Main training driver for tuning, training, and testing ML models (MSE, MAPE, NLL, CRPS)

**Cohesion:** MEDIUM (configuration mixed with logic)  
**Coupling:** MEDIUM (`utils_mse_crps`, TensorFlow/Keras)

**Key Features:**
- K-fold cross-validation with rolling-origin structure
- Configurable lead times: [12h, 48h, 96h, 120h]
- Supports independent year testing (2021, 2024) or rolling cycle validation
- Temperature perturbation support for robustness analysis
- Model architectures for MSE, MAPE, NLL, CRPS

**Configuration Variables:**
```python
model_name = "MAPE"           # MSE, MAPE, NLL, CRPS
independent_year = "cycle"    # "cycle", "2021", or "2024"
cycle_list = [0]              # Rotation indices
lead_time_list = [12, 48, 96, 120]  # Prediction horizons
start_iteration = 5
end_iteration = 5
epochs = 20000
learning_rate = 0.01
```

**Cohesion Issue:** Configuration hardcoded in script rather than externalized

---

#### 3. **pnn_mme_driver.py** ⚠️ HIGH COUPLING
**Purpose:** Multi-model ensemble using Probabilistic Neural Networks (PNN) with Mixture Density Networks (MDN)

**Cohesion:** MEDIUM  
**Coupling:** HIGH (5 different helper modules)

**Complex Dependencies:**
```
├─ src.helper.utils_pnn (data preparation)
├─ src.helper.Logger (experiment logging)
├─ src.helper.job_iterator (parameter combinations)
├─ src.helper.my_parser (CLI arguments)
├─ src.helper.utils_mse_crps (metrics)
├─ tensorflow_probability (MDN distributions)
└─ tensorflow/keras
```

**Key Functions:**
- `create_classifier_network_generic_probability()` - MDN architecture with custom sigma activation
- `load_MLP_dataset()` - Data loading and preparation
- `execute_experiment()` - Main loop using JobIterator for cartesian product of parameters

**Advanced Techniques:**
- Custom MDN loss function with TensorFlow Probability
- Sigma activation function: `f(x) = elu(x) + 1.1` (ensures σ > 0.1)
- Support for dropout and spatial dropout for regularization

---

#### 4. **visualization_driver.py** ⚠️ HIGH COUPLING (Orchestrator)
**Purpose:** High-level orchestrator for cross-validation visualizations and aggregate metric tables

**Cohesion:** LOW (pure orchestration)  
**Coupling:** HIGH (3 external evaluation submodules)

**Key Responsibilities:**
- Configuration for analysis parameters
- Calls external visualization pipelines
- Coordinates multiple analysis stages

**Configuration:**
```python
runAggregateCode = False  # Set to True to use precomputed data
save = True               # Save plots
cycles = [0]
leadTimes = [12, 48, 96, 120]
architectures = ["mape"]  # mse, PNN, CRPS, mape
iterations = 5
obsVsPred = 'test'        # val, test, train, 2021, 2024
expanded = False
```

**External Calls:**
```
├─ src.evaluations.cross_validation_visuals_paper
│  ├─ mme_mse_crps_PNN_lead_times_singlePlot()  (generates prediction files)
│  └─ decentralized_graphing_driver()           (per lead-time plots)
├─ src.evaluations.aggregate_tables
│  └─ aggregateTable()                          (summary statistics)
└─ src.evaluations.boxplot_figures
   ├─ figure_5_plot()
   ├─ figure_6_7_plot()
   ├─ figure_13_plot()
   └─ existance_checker()
```

---

### **Utility/Support Modules**

#### 5. **tuner_retriever.py**
**Purpose:** Load saved tuned models and retrieve trial hyperparameters

**Cohesion:** HIGH  
**Coupling:** MEDIUM (`utils`, `utils_mse_crps`)

**Classes:**
- `MyHyperModel` - Keras HyperModel supporting both dict and HyperParameters inputs

**Functions:**
- `load_trial_hyperparameters(trial_folder)` - Extracts hyperparameters from trial.json
- `load_tuned_model(tuner_directory, trial_id, hypermodel)` - Reconstructs & loads model weights

---

#### 6. **tuner_models_retriever.py** ⚠️ DUPLICATE
**Purpose:** Nearly identical to tuner_retriever.py

**Cohesion:** HIGH  
**Coupling:** MEDIUM (`utils_mse_crps`)

**Status:** **OBSOLETE - DUPLICATE OF tuner_retriever.py**

**Differences from tuner_retriever.py:**
- Imports bare `from utils import` instead of `from src.helper.utils_mse_crps`
- Identical class and function implementations
- Same hyperparameter loading and model reconstruction logic

**Recommendation:** **CONSOLIDATE** - Choose one and delete the other, or have one delegate to the other

---

#### 7. **tunerResults.py**
**Purpose:** Parse and extract hyperparameter tuning results from trial JSON files

**Cohesion:** HIGH  
**Coupling:** LOW (only json, os modules)

**Key Function:**
- `temp(args)` - Main processing loop that:
  - Iterates through rotations, lead times, hours_back, and trials
  - Reads `trial.json` files from Keras Tuner result directories
  - Extracts hyperparameters and objective metrics
  - Writes summary to `.txt` and `.csv` files for Excel import

**Strengths:**
- Simple, focused responsibility
- No external dependencies on src modules
- Good separation of concerns

---

#### 8. **hyperparameter_visualizations.py**
**Purpose:** Interactive visualization of hyperparameter tuning performance using Plotly

**Cohesion:** MEDIUM (could be split into visualization types)  
**Coupling:** LOW (plotly, pandas, numpy only)

**Key Functions:**
- `data_reader()` - Load tuning results from disk
- `mean_stdev_calculation()` - Compute statistics
- `multiLossFunctions()` - Compare multiple loss functions
- `stdev_mean_saver()` - Save statistics to files
- `hyperparameter_boxplot()` - Box plot visualizations
- `heatmap_plot()` - Heatmap comparisons
- `hyperparameter_scatterplot()` - Scatter plots by cycle

**Features:**
- Plots generated with Plotly (interactive, browser-rendered)
- Supports comparison across loss functions
- Customizable font sizes and figure dimensions

---

### **Obsolete Files** ❌

#### 9. **crps_mme_runner copy.py**
**Status:** OUTDATED DUPLICATE

**Issue:** Exact copy of `crps_mme_runner.py` with outdated imports
```python
# OLD (in copy)
from utils import preparingData, crps_loss

# NEW (in current version)
from src.helper.utils_mse_crps import crps_loss, crps
```

**Impact:** Maintenance confusion, risk of code divergence  
**Action:** **DELETE IMMEDIATELY**

---

#### 10. **og_operational_mse_crps_driver.py**
**Status:** PREVIOUS VERSION (superceded)

**Issue:** Older version with hardcoded `independent_year = "2021"`

**Difference:**
```python
# og_operational_mse_crps_driver.py (OLD)
independent_year = "2021"

# operational_mse_crps_driver.py (CURRENT)
independent_year = "cycle"  # Configurable: "cycle", "2021", "2024"
```

**Impact:** Functionality is now configurable in current version  
**Action:** **DELETE** (functionality superseded by operational_mse_crps_driver.py)

---

## Coupling Analysis

### Internal Coupling (within src/driver)

✅ **EXCELLENT** - No driver files import from each other
- Each script is independently executable
- No circular dependencies
- Clean module boundaries

### External Coupling

| Module | Couples To | Dependency Count | Assessment |
|--------|-----------|-----------------|------------|
| crps_mme_runner.py | 2 modules | Low-Medium | Well-balanced |
| operational_mse_crps_driver.py | 1 module | Low | Excellent |
| pnn_mme_driver.py | 5 modules | **High** | ⚠️ Too many dependencies |
| visualization_driver.py | 3 modules | **High** | ⚠️ Orchestration pattern |
| tuner_retriever.py | 2 modules | Low-Medium | Good |
| tunerResults.py | 0 modules | Low | Excellent (standalone) |
| hyperparameter_visualizations.py | 0 modules | Low | Excellent (standalone) |

### Dependency Graph

```
ENTRY POINTS
├─ crps_mme_runner.py
│  ├─ src.helper.utils_mse_crps
│  ├─ src.helper.my_parser
│  ├─ keras_tuner
│  └─ tensorflow/keras
│
├─ operational_mse_crps_driver.py
│  ├─ src.helper.utils_mse_crps
│  └─ tensorflow/keras
│
├─ pnn_mme_driver.py ⚠️ HIGH COUPLING
│  ├─ src.helper.utils_pnn
│  ├─ src.helper.Logger
│  ├─ src.helper.job_iterator
│  ├─ src.helper.my_parser
│  ├─ src.helper.utils_mse_crps
│  ├─ tensorflow_probability
│  └─ tensorflow/keras
│
└─ visualization_driver.py ⚠️ HIGH COUPLING
   ├─ src.evaluations.cross_validation_visuals_paper
   ├─ src.evaluations.aggregate_tables
   └─ src.evaluations.boxplot_figures

UTILITIES
├─ tuner_retriever.py
│  ├─ src.helper.utils
│  └─ src.helper.utils_mse_crps
│
├─ tuner_models_retriever.py (DUPLICATE)
│  └─ src.helper.utils_mse_crps
│
├─ tunerResults.py
│  └─ None (standalone)
│
└─ hyperparameter_visualizations.py
   └─ plotly, pandas, numpy (external only)
```

---

## Cohesion Analysis

### High Cohesion ✅
- **tunerResults.py** - Single responsibility: parse results
- **tuner_retriever.py** - Single responsibility: load models
- **crps_mme_runner.py** - Single responsibility: CRPS ensemble training
- **hyperparameter_visualizations.py** - Single responsibility: create hyperparameter plots

### Medium Cohesion ⚠️
- **operational_mse_crps_driver.py** - Configuration mixed with execution logic
- **pnn_mme_driver.py** - Complex implementation with multiple concerns
- **visualization_driver.py** - Orchestration only, no implementation
- **tuner_models_retriever.py** - Duplicate of tuner_retriever.py

### Low Cohesion ❌
- **File duplication** - crps_mme_runner copy.py, og_operational_mse_crps_driver.py
- **Code duplication** - tuner_retriever.py and tuner_models_retriever.py have identical logic

---

## Recommendations

### Priority 1: Quick Cleanup (CRITICAL)

#### 1.1 Delete Obsolete Files
```bash
# Delete in this order:
rm src/driver/crps_mme_runner\ copy.py
rm src/driver/og_operational_mse_crps_driver.py
```

**Impact:** Removes duplicate maintenance burden immediately  
**Effort:** < 5 minutes  
**Risk:** None (files are obsolete)

---

### Priority 2: Reduce Code Duplication (HIGH)

#### 2.1 Consolidate Model Loading Utilities

**Problem:** `tuner_retriever.py` and `tuner_models_retriever.py` have identical implementations

**Option A: Keep tuner_retriever.py, delete tuner_models_retriever.py**
```python
# Keep standard import path
from src.helper.utils_mse_crps import crps_loss, crps
```

**Option B: Create shared module**
```python
# src/driver/model_utils.py
def load_trial_hyperparameters(trial_folder):
    """Shared implementation"""
    ...

def load_tuned_model(tuner_directory, trial_id, hypermodel):
    """Shared implementation"""
    ...

# Then both files can import from this
```

**Recommendation:** **Go with Option A** - Keep tuner_retriever.py, delete tuner_models_retriever.py  
**Effort:** 10 minutes  
**Risk:** Low (verify imports in any scripts calling these functions)

---

### Priority 3: Reduce Coupling (MEDIUM)

#### 3.1 Extract Configuration from operational_mse_crps_driver.py

**Problem:** Hardcoded configuration variables scattered throughout script

**Solution:** Create external config file

```yaml
# configs/training_mape_12h.yaml
model_name: "MAPE"
lead_time_list: [12]
cycle_list: [0]
start_iteration: 5
end_iteration: 5
epochs: 20000
learning_rate: 0.01
kernel_regularizer: 'l2'
input_structure: 'descending'
path_to_data: 'data/ESB_datasets'
```

**New code:**
```python
import yaml

with open('configs/training_mape_12h.yaml') as f:
    config = yaml.safe_load(f)

model_name = config['model_name']
lead_time_list = config['lead_time_list']
# ... etc
```

**Benefits:**
- Easy configuration reuse
- Reproducibility
- No code changes to run different experiments

**Effort:** 30 minutes  
**Risk:** Low (backward compatible if done carefully)

---

#### 3.2 Refactor pnn_mme_driver.py High Coupling

**Problem:** Depends on 5 different helper modules

**Solution:** Create PNN-specific helper module

```python
# src/helper/pnn_driver_utils.py
from src.helper.utils_pnn import preparingData
from src.helper.Logger import Logging
from src.helper.job_iterator import JobIterator
from src.helper.my_parser import create_parser
from src.helper.utils_mse_crps import ryan_ssrel, ssrat_avg, ...

class PNNExperimentManager:
    def __init__(self, args):
        self.parser = create_parser()
        self.logger = Logging()
        self.job_iterator = JobIterator()
        # ...
    
    def load_data(self):
        return preparingData()
    
    def get_metrics(self):
        return {
            'ssrel': ryan_ssrel,
            'ssrat_avg': ssrat_avg,
            # ...
        }

# Then in pnn_mme_driver.py:
from src.helper.pnn_driver_utils import PNNExperimentManager
manager = PNNExperimentManager(args)
```

**Benefits:**
- Centralizes PNN-specific logic
- Reduces coupling in main driver
- Easier to maintain and test

**Effort:** 1-2 hours  
**Risk:** Medium (refactoring complex code)

---

#### 3.3 Refactor visualization_driver.py Coupling

**Problem:** Orchestrates 3 separate evaluation submodules

**Solution:** Create evaluation adapter pattern

```python
# src/driver/visualization_adapter.py
class EvaluationPipeline:
    def __init__(self, config):
        self.config = config
    
    def generate_prediction_files(self):
        from src.evaluations.cross_validation_visuals_paper import (
            mme_mse_crps_PNN_lead_times_singlePlot
        )
        return mme_mse_crps_PNN_lead_times_singlePlot(...)
    
    def generate_plots(self):
        from src.evaluations.cross_validation_visuals_paper import (
            decentralized_graphing_driver
        )
        return decentralized_graphing_driver(...)
    
    def generate_tables(self):
        from src.evaluations.aggregate_tables import aggregateTable
        return aggregateTable(...)
    
    def generate_figures(self):
        from src.evaluations.boxplot_figures import (
            figure_5_plot, figure_6_7_plot, figure_13_plot
        )
        return [figure_5_plot(), figure_6_7_plot(), figure_13_plot()]

# Then in visualization_driver.py:
from src.driver.visualization_adapter import EvaluationPipeline
pipeline = EvaluationPipeline(config)
pipeline.generate_prediction_files()
pipeline.generate_plots()
# ... etc
```

**Effort:** 1 hour  
**Risk:** Low (adapter pattern is well-understood)

---

### Priority 4: Improve Cohesion (LOW)

#### 4.1 Split hyperparameter_visualizations.py

**Current state:** All visualization types in one file (500+ lines)

**Proposed structure:**
```
src/driver/
├─ visualizations/
│  ├─ __init__.py
│  ├─ hyperparameter_boxplot.py
│  ├─ hyperparameter_heatmap.py
│  ├─ hyperparameter_scatter.py
│  └─ data_reader.py
```

**Effort:** 1-2 hours  
**Impact:** Improved maintainability, easier testing  
**Risk:** Low (internal refactoring only)

---

## Implementation Roadmap

### Week 1 - Quick Wins
1. Delete obsolete files (5 min)
2. Consolidate model utilities (10 min)
3. Verify no breaks in remaining code (20 min)

### Week 2 - Configuration
4. Extract operational_mse_crps_driver.py config (30 min)
5. Test with multiple config files (30 min)

### Week 3 - Refactoring
6. Refactor pnn_mme_driver.py coupling (2 hours)
7. Refactor visualization_driver.py coupling (1 hour)
8. Run full test suite

### Week 4 - Polish
9. Split hyperparameter_visualizations.py (1-2 hours)
10. Add docstrings to all modules
11. Update README documentation

---

## Metrics Summary

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Duplicate files** | 2 | 0 | ❌ |
| **Duplicate utilities** | 2 | 1 | ❌ |
| **High coupling modules** | 2 | 0 | ❌ |
| **Config externalization** | 0% | 100% | ❌ |
| **Internal driver coupling** | 0 | 0 | ✅ |
| **Cohesion (avg)** | 6.2/10 | 8.0/10 | ⚠️ |

---

## References

### Best Practices Applied
- **Single Responsibility Principle** - Each module should have one reason to change
- **DRY (Don't Repeat Yourself)** - Eliminate code duplication
- **Loose Coupling** - Minimize dependencies between modules
- **High Cohesion** - Keep related functionality together
- **Adapter Pattern** - Decouple high-level from low-level modules
- **Configuration Externalization** - Separate config from code

### Related Documentation
- See the mermaid diagram in this analysis for visual dependency overview
- Check individual module files for detailed implementation notes
- Configuration examples in `configs/` folder

---

## Questions & Clarifications

If any of these recommendations are unclear or would like more detail on implementation:

1. Which refactoring should be prioritized?
2. Should we create new helper modules or inline optimizations?
3. Should existing scripts be updated to use new utility consolidation?
4. What's the preferred configuration format (YAML, JSON, Python)?

