#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include "tsp.h"

using namespace std;

const int THREADS_PER_BLOCK = 256;

//forward delcarations
int get_nearest_neighbor(int, bool[]);
__global__ void compute_distances(int, int, int, int*, int*, int*);


//global variables
int num_nodes, num_array_bytes;
int num_devices;
int *x, *y, *dist;
int **dev_x, **dev_y, **dev_dist;
set<edge> used_edges;

int main(int argc, char* argv[]){

    //get number of nodes before reading coords, 
    //as dynamic arrays must already be sized.
    num_nodes = get_num_nodes(argv[1]);
    num_array_bytes = num_nodes*sizeof(int);

    //coordinates and distances on host
    x = new int[num_nodes];
    y = new int[num_nodes];
    dist = new int[num_nodes];

    //coordinates and distances on each device
    dev_x = new int*[num_devices];
    dev_y = new int*[num_devices];
    dev_dist = new int*[num_devices];

    //get id, x & y values from file
    read_coords(argv[1], num_nodes, x, y);

    //allocate memory on device and tranfer coordinates
    cudaGetDeviceCount(&num_devices);
    for(int i=0; i<num_devices; ++i){
        cudaSetDevice(i);

        cudaMalloc( (void**)&dev_x[i], num_array_bytes);
        cudaMalloc( (void**)&dev_y[i], num_array_bytes);
        cudaMalloc( (void**)&dev_dist[i], num_array_bytes);

        cudaMemcpy(dev_x[i], x, num_array_bytes, cudaMemcpyHostToDevice );
        cudaMemcpy(dev_y[i], y, num_array_bytes, cudaMemcpyHostToDevice );
    }

    //initialize used node arrays
    bool used1[num_nodes], used2[num_nodes];
    for(int i=0; i<num_nodes; ++i){
        used1[i] = used2[i] = false;
    }

    //take the first points from each path at random
    srand(time( NULL ));
    int id1 = rand() % num_nodes;
    used1[id1] = true;
    int id2 = rand() % num_nodes;
    used2[id2] = true;
    cout << "path1\tpath2" << endl;
    cout << id1 << "\t" << id2 << endl;
    
    //for each path, add nodes using nearest neighbor algorithm
    for(int i=1; i<num_nodes; ++i){
        id1 = get_nearest_neighbor(id1, used1);
        id2 = get_nearest_neighbor(id2, used2);
        cout << id1 << "\t" << id2 << endl;
    }

    for(int i=0; i<num_devices; ++i){
        cudaFree( dev_x[i] );
        cudaFree( dev_y[i] );
        cudaFree( dev_dist[i] );
    }
}

/* Gets nearest neighbor by computing the distance to all other
 * nodes (in parrallel on the GPU), then iterating the distances
 * and noting the closest available to return.  Deterministic.
 */
int get_nearest_neighbor(int start, bool used[]){

    if( DEBUG ){
        cout << "Computing for node: " << start << endl;
        cout << "Used nodes: ";
        for(int i=0; i<num_nodes; ++i){
            used[i] && cout << i << " ";
        }
        cout << endl;
    }

    //split workload evenly amongst devices - each device responsible for a 
    //congruant block of distances.
    int blocks_per_grid = (num_nodes + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
    int blocks_per_device = blocks_per_grid / num_devices;
    if( blocks_per_device == 0) blocks_per_device = 1;
    int nodes_per_device = ceil(static_cast<double>(num_nodes) / num_devices);
    int offset, offset_end;

    for(int i=0; i<num_devices; ++i){
        offset = i*nodes_per_device;
        cudaSetDevice(i);
        if(DEBUG) cout << "Runnin compute_distance, offest: " << offset
                        << ", device: " << i 
                        << ", blocks: " << blocks_per_device 
                        << endl;

        compute_distances<<<blocks_per_device,THREADS_PER_BLOCK>>>(start, offset, num_nodes, dev_x[i], dev_y[i], dev_dist[i]);
    }

    int low_id = -1;//-1 used as a sentinel to check valid id found
    int low_dist;
    bool first = true;

    //iterate of distances computd by each device
    for(int i=0; i<num_devices; ++i){

        offset = i*nodes_per_device;
        offset_end = offset + nodes_per_device;

        cudaSetDevice(i);
        cudaMemcpy(dist, dev_dist[i], num_array_bytes, cudaMemcpyDeviceToHost );

        //find the lowest distance from this block per card.  the offset_end
        //may round up to be higher then the number of nodes, so check that.
        for(int end=offset; end<offset_end && end <num_nodes; ++end){

            if(DEBUG2) cout << end << ":" << dist[end] << " ";

            //if this node has been used in this path
            if(used[end] || 
               start == end ){ continue; }
            
            //if this edge has already been used
            if( used_edges.find(edge(start,end)) != used_edges.end()){
                continue;
            }

            //unused path to a different node, use it.
            if( first ) {
                low_id = end;
                low_dist = dist[end];
            } else {
                if( dist[end] < low_dist ){
                    low_id = end;
                    low_dist = dist[end];
                }
            }
            first = false;
        }
        if(DEBUG2) cout << endl;
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

//since we're only interested in the relative distances, we don't need
//to compute the square root in the pythagorean theorem.
__global__ void compute_distances(int start, int offset, int num_nodes, int *x, int *y, int *dist){
    int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;
    if( tid < num_nodes ){
        dist[tid] = (x[tid]-x[start])*(x[tid]-x[start]) + (y[tid]-y[start])*(y[tid]-y[start]);
    }
}

