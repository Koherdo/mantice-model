
**docs/theory.md**
```markdown
# Theoretical Background

## Mathematical Framework

### Quaternionic Calendars
Each node has a quaternionic state:
$$ \mathbb{C}_i(t) = C^0_i(t) + C^1_i(t)\mathbf{i} + C^2_i(t)\mathbf{j} + C^3_i(t)\mathbf{k} $$

### Synchronization Dynamics
$$ \frac{d\mathbb{C}_i}{dt} = \boldsymbol{\Omega}_i + \frac{\sigma}{k_i} \sum_{j \in \mathcal{N}(i)} \mathcal{F}(\mathbb{C}_j, \mathbb{C}_i, \mathbf{r}_{ij}) + \boldsymbol{\Xi}_i(t) $$

## Phase Transitions
The system exhibits a second-order phase transition at critical coupling $\sigma_c$ with exponent $\beta = 1/2$.