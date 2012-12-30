#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "book.h"
#include "tsp.h"

using namespace std;

const int THREADS_PER_BLOCK = 256;

//forward delcarations
int get_nearest_neighbor(int, bool[]);
__global__ void compute_distances(int, int, int*, int*, int*);


//global variables
int num_nodes;
int *x, *y, *dist;
int *dev_x, *dev_y, *dev_dist;
set<edge> used_edges;

int main(int argc, char* argv[]){

    num_nodes = get_num_nodes(argv[1]);

    x = new int[num_nodes];
    y = new int[num_nodes];
    dist = new int[num_nodes];
    read_coords(argv[1], num_nodes, x, y);

    HANDLE_ERROR( cudaMalloc( (void**)&dev_x, num_nodes*sizeof(int)));
    HANDLE_ERROR( cudaMalloc( (void**)&dev_y, num_nodes*sizeof(int)));
    HANDLE_ERROR( cudaMalloc( (void**)&dev_dist, num_nodes*sizeof(int)));

    HANDLE_ERROR( cudaMemcpy(dev_x, x, num_nodes*sizeof(int), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy(dev_y, y, num_nodes*sizeof(int), cudaMemcpyHostToDevice ) );

    bool used1[num_nodes], used2[num_nodes];
    for(int i=0; i<num_nodes; ++i){
        used1[i] = false;
        used2[i] = false;
    }

    //srand(time( NULL ));
    srand(1);
    int id1 = rand() % num_nodes;
    used1[id1] = true;
    int id2 = rand() % num_nodes;
    used2[id2] = true;
    cout << "path1\tpath2" << endl;
    cout << id1 << "\t" << id2 << endl;
    
    for(int i=1; i<num_nodes; ++i){
        id1 = get_nearest_neighbor(id1, used1);
        id2 = get_nearest_neighbor(id2, used2);
        cout << id1 << "\t" << id2 << endl;
    }

    cudaFree( dev_x );
    cudaFree( dev_y );
    cudaFree( dev_dist );
}

int get_nearest_neighbor(int start, bool used[]){

    const int BLOCKS_PER_GRID = (num_nodes + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;

    compute_distances<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(start, num_nodes, dev_x, dev_y, dev_dist);
    HANDLE_ERROR( cudaMemcpy(dist, dev_dist, num_nodes*sizeof(int), cudaMemcpyDeviceToHost ) );

    int low_id = -1;
    bool first = true;

    if( DEBUG ){
        cout << "Computing for node: " << start << endl;
        cout << "Used nodes: ";
        for(int i=0; i<num_nodes; ++i){
            used[i] && cout << i << " ";
        }
        cout << endl;
    }

    for(int end=0; end<num_nodes; ++end){
        if(used[end] || 
           start == end ){ continue; }
        
        //if this path has already been used
        edge current_edge(start, end);
        if( used_edges.find(current_edge) != used_edges.end()){
            continue;
        }

        //unused path to a different node, use it.
        if( first ) {
            low_id = end;
        } else {
            if( dist[end] < dist[low_id] ){
                low_id = end;
            }
        }
        first = false;
    }
    if( low_id == -1 ){
        cout << "ERROR:reached point were all valid paths used." << endl;
        exit(1);
    }

    if(DEBUG) cout << "Lowest Valid: " << low_id << endl;
    used_edges.insert(edge(start, low_id));
    used[low_id] = true;
    return low_id;
}

__global__ void compute_distances(int start, int num_nodes, int *x, int *y, int *dist){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if( tid < num_nodes ){
        dist[tid] = sqrtf(
            (float)((x[tid]-x[start])*(x[tid]-x[start]) + (y[tid]-y[start])*(y[tid]-y[start]))
        );
    }
}

