# 🎓 Training Parameters Explanation

## 📋 Parameters

### 1. **Model Type**

#### CNN Model
- **What does it do?** Analyzes a single signature
- **Output:** Is the signature genuine (1) or forged (0)?
- **Usage:** To check a single signature
- **Advantage:** Fast, simple, only one signature needed
- **Example:** "Is this signature genuine?"

#### Siamese Network
- **What does it do?** Compares two signatures
- **Output:** Do the two signatures belong to the same person?
- **Usage:** To compare two signatures
- **Advantage:** More sensitive comparison
- **Example:** "Do these two signatures belong to the same person?"

---

### 2. **Epochs (Training Cycle Count)**

**What does it mean?**
- How many times the model will see the entire dataset
- The model learns and improves with each epoch

**Recommended Values:**
- **Beginner:** 20-30 (for quick testing)
- **Normal:** 50-100 (for good results)
- **Advanced:** 100-200 (for best results, takes longer)

**Note:**
- Too few epochs → Model cannot learn enough (underfitting)
- Too many epochs → Model memorizes, cannot generalize (overfitting)
- Early Stopping automatically stops (if no improvement for 10 epochs)

**Example:**
```
Epochs = 50
→ Model will see the dataset 50 times
→ Will learn better each time
→ May take approximately 1-3 hours (depending on data amount)
```

---

### 3. **Batch Size (Batch Processing Size)**

**What does it mean?**
- How many signatures will be processed together at once
- Adjusted according to GPU/CPU memory

**Recommended Values:**
- **Small dataset (<1000 signatures):** 16-32
- **Medium dataset (1000-5000):** 32-64
- **Large dataset (>5000):** 64-128

**Memory Usage:**
- **Batch Size = 16** → Low memory, slow
- **Batch Size = 32** → Medium memory, balanced (recommended)
- **Batch Size = 64** → High memory, fast
- **Batch Size = 128** → Very high memory, very fast

**Note:**
- Too small → Training becomes slow
- Too large → Memory error (Out of Memory) may occur
- If GPU is available, larger values can be used

**Example:**
```
Batch Size = 32
→ 32 signatures processed at once
→ If there are 1000 signatures → 1000/32 = 31.25 → 32 batches
→ 32 batches processed per epoch
```

---

## 🔧 Parameter Combinations

### Quick Test (Quick Trial)
```
Model Type: CNN
Epochs: 20
Batch Size: 32
Duration: ~15-30 minutes
Result: Fast but low accuracy
```

### Balanced (Recommended)
```
Model Type: CNN
Epochs: 50
Batch Size: 32
Duration: ~1-2 hours
Result: Good accuracy (≥90% target)
```

### Best Result (Long Duration)
```
Model Type: CNN
Epochs: 100
Batch Size: 32
Duration: ~2-4 hours
Result: Highest accuracy
```

### If GPU Available (Fast + Good)
```
Model Type: CNN
Epochs: 50
Batch Size: 64 or 128
Duration: ~30-60 minutes
Result: Fast and good accuracy
```

---

## 📊 Parameter Effects

### If Epochs Increase:
- ✅ Better learning
- ✅ Higher accuracy
- ❌ Longer duration
- ⚠️ Overfitting risk (memorization)

### If Batch Size Increases:
- ✅ Faster training
- ✅ More stable learning
- ❌ More memory usage
- ⚠️ Memory error risk

---

## 🎯 Recommended Starting Settings

**First Trial:**
- Model Type: **CNN**
- Epochs: **30**
- Batch Size: **32**

**Normal Usage:**
- Model Type: **CNN**
- Epochs: **50**
- Batch Size: **32**

**Best Result:**
- Model Type: **CNN**
- Epochs: **100**
- Batch Size: **32**

---

## ⚙️ Automatic Features

The training in GUI has the following features:

1. **Early Stopping**
   - Stops if no improvement for 10 epochs
   - Saves the best model

2. **Learning Rate Reduction**
   - Reduces learning rate if improvement stops
   - Performs finer tuning

3. **Model Checkpointing**
   - Saves the best model at each epoch
   - Best model is preserved even if training is interrupted

4. **Data Augmentation**
   - Rotates, shifts, zooms signatures
   - Acts like more data
   - Prevents overfitting

---

## 💡 Tips

1. **If training for the first time:**
   - Start with Epochs: 20-30
   - See the results
   - Increase if needed

2. **If you get a memory error:**
   - Reduce Batch Size (32 → 16)
   - Or use less data

3. **If it takes too long:**
   - Reduce Epochs
   - Increase Batch Size (if GPU available)

4. **If accuracy is low:**
   - Increase Epochs
   - Add more data
   - Data augmentation is already active

---

## 📈 Expected Results

### For CNN Model:
- **20 Epochs:** ~70-80% accuracy
- **50 Epochs:** ~85-92% accuracy (target)
- **100 Epochs:** ~90-95% accuracy

### For Siamese Network:
- **20 Epochs:** ~75-85% accuracy
- **50 Epochs:** ~88-93% accuracy
- **100 Epochs:** ~90-95% accuracy

---

## 🔍 Parameter Selection Table

| Data Amount | Epochs | Batch Size | Estimated Duration |
|-------------|--------|------------|-------------------|
| < 500 signatures | 30 | 16 | 15-30 min |
| 500-2000 | 50 | 32 | 1-2 hours |
| 2000-5000 | 50-100 | 32-64 | 2-4 hours |
| > 5000 | 100 | 64-128 | 4-8 hours |

---

## ❓ Frequently Asked Questions

**Q: How many epochs should I use?**
A: 50 for starting, 100 for best results.

**Q: What should Batch Size be?**
A: 32 works well in most cases. If you get a memory error, reduce to 16.

**Q: Which model should I choose?**
A: CNN is recommended for starting. You can try Siamese later.

**Q: How long does training take?**
A: Depends on data amount and parameters. Usually 1-4 hours.

**Q: Can I turn off the computer during training?**
A: No! If training is interrupted, you need to start over. The best model is automatically saved but training won't complete.

**Q: What is overfitting?**
A: The model memorizing the data. Early Stopping prevents this.

---

## 🎓 Summary

- **Model Type:** CNN (for starting) or Siamese (for comparison)
- **Epochs:** 50 (balanced) or 100 (best)
- **Batch Size:** 32 (ideal for most cases)

**Simple Rule:** Start with 50 epochs, 32 batch size for first trial! 🚀
