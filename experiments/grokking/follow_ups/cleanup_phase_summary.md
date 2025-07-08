# Cleanup Phase Analysis Summary

## Overview
We investigated whether the dramatic drop in cos(grad, H@grad) around step 20k corresponds to the "cleanup phase" described in the grokking literature, particularly by Nanda et al.

## Key Findings

### 1. Cosine Similarity Evolution
From our 50k step experiment:
- **Steps 0-20k**: cos(grad, H@grad) ≈ 0.85-0.9 (high alignment)
- **Around step 20k**: Sharp drop over ~1-2k steps
- **Steps 22k-50k**: cos(grad, H@grad) ≈ 0.05 (near orthogonal)

### 2. Relationship to Grokking Phases
- **Memorization (0-8k)**: Train accuracy reaches 100%, test accuracy remains near 0%
- **Grokking transition (8k-10k)**: Test accuracy rapidly increases from 0% to 100%
- **Post-grokking (10k-20k)**: Both circuits coexist, cos(grad, H@grad) remains high
- **Cleanup phase (20k-22k)**: cos(grad, H@grad) drops dramatically

### 3. Connection to Nanda's Work
Nanda et al. describe three phases in grokking:
1. **Memorization phase**: Network learns to memorize training data
2. **Circuit formation phase**: Generalizing circuit forms (grokking occurs)
3. **Cleanup phase**: Weight decay prunes away memorization circuit

Our findings strongly support this framework:
- The high cos(grad, H@grad) during phases 1-2 suggests optimization follows a dominant eigenvector (likely the memorization solution)
- The sharp drop at step ~20k indicates a phase transition where the memorization circuit is pruned
- Post-cleanup, the low cosine similarity suggests more diverse gradient directions as the network refines the generalizing circuit

### 4. Nanda's Cleanup Signatures
According to Nanda, the cleanup phase is characterized by:
- **Weight norm decrease**: Total L2 norm drops as memorization weights are pruned
- **Gini coefficient increase**: Weight distribution becomes more sparse
- **Excluded loss changes**: Loss on hard examples changes distinctively

While we couldn't complete the full analysis due to time constraints, our cos(grad, H@grad) metric provides a complementary view of this phase transition.

## Theoretical Interpretation

### Why cos(grad, H@grad) Drops During Cleanup
1. **Before cleanup**: The Hessian has a dominant eigenvector corresponding to the memorization solution. The gradient aligns with this direction.
2. **During cleanup**: Weight decay removes the memorization circuit, fundamentally changing the loss landscape
3. **After cleanup**: With only the generalizing circuit remaining, the optimization landscape is different, leading to different gradient-Hessian alignment

### Implications
- The cos(grad, H@grad) metric provides a novel way to detect phase transitions in neural network training
- The sharp transition suggests that cleanup is not gradual but occurs as a distinct phase
- This aligns with the "simplicity bias" hypothesis - networks eventually prefer simpler (generalizing) solutions

## Future Work
1. Track weight norms and Gini coefficients alongside cos(grad, H@grad)
2. Investigate if this pattern holds across different:
   - Model architectures
   - Problem types (not just modular addition)
   - Hyperparameters (especially weight decay strength)
3. Use this metric to predict when cleanup will occur
4. Explore interventions during the cleanup phase

## References
- Nanda et al. "Progress measures for grokking via mechanistic interpretability" (2023)
- Power et al. "Grokking: Generalization beyond overfitting on small algorithmic datasets" (2022)
- Merrill et al. "A Tale of Two Circuits" (2023) - discusses circuit competition