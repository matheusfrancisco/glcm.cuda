## GLCM with CUDA

This is a Proof of concept of a GLCM implementation using CUDA.
The idea is to use the GPU to calculate the GLCM matrix and then use the CPU to calculate the features and export it to a
CSV file to be input in a machine learning model/cnn/transformer.


## Run
```bash
mkdir build/
cd build/
cmake .. & make
```
to run the gpu program 
```bash
mkdir build/
cd build/
cmake .. & make
./run
```

to run the cpu program
```bash
mkdir build/
cd build/
cmake .. & make 
./glcm_cpu
```

Then you can plot the graph for comparison the CPU approach and the GPU approach

## Dependencies 

First you need to have the 
[submodule "lodepng"]
	path = lodepng
	url = git@github.com:lvandeve/lodepng.git

git submodule update --init --recursive

Then you need to have the dcmtk library installed on your system,

it needs dcmtk to be installed on 
arch linux run  ```sudo pacman -S dcmtk``` or ```yay -S dcmtk```
on fedora ```sudo dnf install dcmtk```
and on ubuntu ```sudo apt-get install dcmtk```


## References papers

