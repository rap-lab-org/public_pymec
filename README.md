# A Probabilistic Measure of Multi-Robot Connectivity and Ergodic Optimal Control

## 📦 Installation  

### Conda Environment Setup  
```bash
# Clone repository  
git clone https://github.com/rap-lab-org/public_pymec  
cd public_pymec 

# Create and activate conda environment (Python 3.12 recommended)  
# Note that the results may vary slightly due to the version of JAX
conda create -n imec -f imec.yml 
conda activate imec  
```
---

## 🧪 Toy Example  

▶️ Refer to **[imec.ipynb](./imec.ipynb)** and **[baseline.ipynb](./baseline.ipynb)** for tutorial.  

---

## ✨ Citation  

```bibtex
@article{liu2025probabilistic,
    title={A Probabilistic Measure of Multi-Robot Connectivity and Ergodic Optimal Control}, 
    author={Yongce Liu, Zhongqiang Ren},
    journal={RSS},
    year={2025},
}
```

---

## 📚 Acknowledgement  

This project builds upon the following works:  

### iLQR Implementation  
```bibtex
@article{sun2025fast,
  title={Fast Ergodic Search With Kernel Functions},
  author={Sun, Max Muchen and Gaggar, Ayush and Traumore},
  journal={IEEE Transactions on Robotics},
  year={2025}
}
```  
[ergodic-control-sandbox: https://github.com/MurpheyLab/ergodic-control-sandbox](https://github.com/MurpheyLab/ergodic-control-sandbox)  

### Modification
- The codebase was optimized with JAX's computational primitives for improved performance and simplicity.

### Time-Optimal Ergodic Search  
```bibtex
@article{dong2023TimeOptErg,
    title={Time-Optimal Ergodic Search}, 
    author={Dong, Dayi and Berger, Henry and Abraham, Ian},
    journal={RSS},
    year={2023}
}
```  
[time_optimal_ergodic_search: https://github.com/ialab-yale/time_optimal_ergodic_search](https://github.com/ialab-yale/time_optimal_ergodic_search)  

<!-- ## Note
jax[cpu] and jax[gpu], results may be different. -->