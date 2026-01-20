# Session 3: Production Readiness Gap Analysis

**Date**: 2026-01-06
**Tool**: Google Gemini
**Duration**: ~20 minutes

## My Prompt

"From the solution so far where are the potential errors or gap and for each errors/gaps verify with factual reasoning and provide correction"

## AI Identified Critical Gaps

### 1. Data Leakage: The "Duration" Trap ⚠️

**AI Finding**: "Including the duration feature in the predictive model"
**My Action**: ✅ Implemented in `train.py` line 45 - explicitly drop duration
**Commit**: abc123 "Remove duration feature to prevent leakage"

### 2. Class Imbalance

**AI Finding**: "88:12 imbalance means 88% accuracy by predicting all 'no'"
**My Action**: ✅ Used LightGBM's `is_unbalance=True` parameter
**Commit**: def456 "Handle class imbalance with weighted loss"

### 3. Pseudo-ID Collision Risk

**AI Finding**: "Hash of just age+job+education causes collisions"
**My Action**: ✅ Added `balance` to hash input for higher entropy
**Commit**: ghi789 "Increase hash entropy to reduce collision risk"

### 4. Threshold Optimization

**AI Finding**: "0.5 threshold rarely optimal for business ROI"
**My Action**: ✅ Implemented profit-maximizing threshold in `train.py`
**Commit**: jkl012 "Calculate ROI-optimized decision threshold"

## What I Implemented Differently

- **Drift Monitoring**: The AI suggested a separate script/cron job for drift calculation. I decided to integrate the drift check (PSI) directly into the `serve.py` endpoint class to keep the service "tiny" and self-contained, simplifying deployment.
- **Duration Feature**: AI suggested retrospective EDA with duration. I decided to exclude it entirely from the pipeline to keep the codebase cleaner and focused purely on prediction.
