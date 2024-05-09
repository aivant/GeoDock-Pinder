### install geodock env on mac-silicon

```mamba create -n geodock python=3.10```

```mamba activate geodock```

```pip install -r requirements_cpu.txt```

install the "special cases" by hand (sorry):

```mamba install -c conda-forge pdbfixer```

```pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html ```

```pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cpu.html```

```pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html```

``` pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html```

```pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html```

```pip install "git+https://github.com/facebookresearch/pytorch3d.git"```

```pip install -e .```

try then environment by running `python geodock/GeoDockRunner.py`. This will dock an example dimer, creating a `test.pdb` file at the root of the repo.
### notes
- `triton` package removed from requirements_cpu.txt, I couldn't find it anywhere...

