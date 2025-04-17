# CoupledQuantumSystems

![gpu_meme](assets/gpu_meme.jpg)

Quality of life improvements for quantum simulations,
- [ ] GPU simulation chekcpointing
- [ ] Product basis - dressed basis assignment and fast conversion
- [ ] a lot more
 
```
pip3 install git+https://github.com/JiakaiW/CoupledQuantumSystems
```

to use it with dynamiqs, you need jax with GPU, e.g.:
'''
pip install --upgrade "jax[cuda12]"
'''

![workflow of quantum simulation using this package](assets/CoupledQuantumSystems.png)
