## GLCM with CUDA




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


[submodule "lodepng"]
	path = lodepng
	url = git@github.com:lvandeve/lodepng.git


## References papers

