all:nearest_neighbor refine_path 2_opt verify
nearest_neighbor:
	nvcc --compiler-options='-Wall' -O3 -o bin/nearest_neighbor nearest_neighbor.cu
refine_path:
	g++ --std=c++0x -Wall -O3 -o bin/refine_path refine_path.cpp
2_opt:
	g++ --std=c++0x -Wall -O3 -o bin/2_opt 2_opt.cpp
verify:
	g++ --std=c++0x -Wall -O3 -o bin/verify verify.cpp
