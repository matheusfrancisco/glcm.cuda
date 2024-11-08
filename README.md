## GLCM with CUDA

Working in progress



## Run e.g
GPU command
```bash
nvcc -arch=sm_70 main.cu lodepng/lodepng.cpp file.cpp image.cpp glcm.cpp -o out --run
```

CPU
```bash
nvcc -arch=sm_70 main_cpu.cpp lodepng/lodepng.cpp file.cpp image.cpp glcm.cpp -o out --run
```

## Dependencies 

it needs dcmtk to be installed on arch linux run  ```sudo pacman -S dcmtk``` or ```yay -S dcmtk```
or on fedora ```sudo dnf install dcmtk```
or on ubuntu ```sudo apt-get install dcmtk```


[submodule "lodepng"]
	path = lodepng
	url = git@github.com:lvandeve/lodepng.git


## References papers

