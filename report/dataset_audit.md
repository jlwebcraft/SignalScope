# SignalScope Official Dataset Audit Report
**Dataset Root**: `C:\Programming\SignalScope-data`  
**Audit Status**: Verified  

---

## 1. Top-Level Structure
- **Subdirectories**: `test`, `train`
- **Metadata/Root Files**: None (labels encoded purely by folder hierarchy)

---

## 2. Training Partition (`train/`)
- **Total Training Images**: 100,000
### Class `FAKE`
- **Count**: 50,000
- **Extensions**: {'.jpg': 50000}
- **Sample Filenames**: `['1000 (10).jpg', '1000 (2).jpg', '1000 (3).jpg', '1000 (4).jpg', '1000 (5).jpg']`
- **Formats Sampled**: {'JPEG': 500}
- **Color Modes**: {'RGB': 500}
- **Dimensions**: Min `32x32`, Max `32x32`
- **Common Dimensions**: [{'dimension': '32x32', 'count': 500}]
- **Corrupt Images**: 0

### Class `REAL`
- **Count**: 50,000
- **Extensions**: {'.jpg': 50000}
- **Sample Filenames**: `['0000 (10).jpg', '0000 (2).jpg', '0000 (3).jpg', '0000 (4).jpg', '0000 (5).jpg']`
- **Formats Sampled**: {'JPEG': 500}
- **Color Modes**: {'RGB': 500}
- **Dimensions**: Min `32x32`, Max `32x32`
- **Common Dimensions**: [{'dimension': '32x32', 'count': 500}]
- **Corrupt Images**: 0

**Class Balance**: Perfectly balanced (1.00:1)  
**Duplicate Filenames across classes**: 40000

---

## 3. Test Partition (`test/`)
- **Total Test Images**: 20,000
### Class `FAKE`
- **Count**: 10,000
- **Extensions**: {'.jpg': 10000}
- **Sample Filenames**: `['0 (10).jpg', '0 (2).jpg', '0 (3).jpg', '0 (4).jpg', '0 (5).jpg']`
- **Common Dimensions**: [{'dimension': '32x32', 'count': 500}]

### Class `REAL`
- **Count**: 10,000
- **Extensions**: {'.jpg': 10000}
- **Sample Filenames**: `['0000 (10).jpg', '0000 (2).jpg', '0000 (3).jpg', '0000 (4).jpg', '0000 (5).jpg']`
- **Common Dimensions**: [{'dimension': '32x32', 'count': 500}]

### Test Partition Role & Strict Exclusion Policy
> [!IMPORTANT]
> **Role of `test/`**: The `test/` directory contains evaluation data. In strict compliance with the SIH 2026 problem requirements and anti-leakage protocol, **the `test/` directory is completely excluded from model training, validation, threshold tuning, and feature selection**.
> All model development, hyperparameter selection, and validation will proceed strictly on local stratified splits constructed from `train/`.

---

## 4. Generator & Provenance Metadata
- **Generator Tracking**: None detected from filenames or subfolders; labels are binary REAL / FAKE.
- **Folder Label Representation**: `0 = REAL`, `1 = FAKE`
