//#include <bitset>
#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <set>
//#include <string>
#include <vector>
//#include "book.h"
#include "tsp.h"

using namespace std;

//used to emulate useful c++11 functions w/o c++11
//list<int>::iterator prev(list<int>::iterator pos){ return --pos; }
//list<int>::iterator next(list<int>::iterator pos){ return ++pos; }


//forward delcarations
void refactor_path(list<int>&, set<edge>&, set<edge>&, set<int>&);
int find_worst_pos(list<int>&, list<int>::iterator&, const set<edge>&, set<int>&);
int find_best_insert(const list<int>::iterator&, list<int>&, list<int>::iterator&, const set<edge>&);


//Global variables.
int num_nodes;
set<edge> used_edges1, used_edges2; //requies fast lookup, rarely modified
list<int> path1, path2; //reequires sequential access and ease of modification
int *x, *y; //will be dynamic arraysr. equires fast random acces, never modified


int main(int argc, char* argv[]){

    num_nodes = get_num_nodes(argv[1]);

    x = new int[num_nodes];
    y = new int[num_nodes];
    read_coords(argv[1], num_nodes, x, y); //three column list of id, x coord, y coord
    read_paths(argv[2], used_edges1, used_edges2, path1, path2); //two column list of points on two paths

    int num_iterations = num_nodes/10; //attempt to refactor worst 10%
    set<int> path1_refactored, path2_refactored;
    for(int i=0; i<num_iterations; ++i){
        refactor_path(path1, used_edges1, used_edges2, path1_refactored);
        refactor_path(path2, used_edges2, used_edges1, path2_refactored);
    }

    cout << "path1\tpath2" << endl;
    list<int>::iterator pos1 = path1.begin();
    list<int>::iterator pos2 = path2.begin();
    while(pos1 != path1.end() && pos2 != path2.end()){
        cout << *pos1 << "\t" << *pos2 << endl;
        ++pos1;
        ++pos2;
    }

}


/* refactor_path() - iterate through the path and find the node (high_pos) with the longest 
 * adjacent edges (that hasn't been rafactored), then iterate through all paths again and
 * find the edge with two nodes closest to the high node as the best place to insert the
 * high node.  Check that new paths created (by both removing and inserting the high node)
 * are not used by the other path. Currently contains a lot of duplicate distance calculations,
 * this is a known area for optimization.
 */
void refactor_path(list<int> &path, set<edge> &edges, set<edge> &other_edges, set<int> &refactored_ids){

    list<int>::iterator high_pos, insert_pos;

    int distance_removed = find_worst_pos(path, high_pos, other_edges, refactored_ids);

    int distance_added = find_best_insert(high_pos, path, insert_pos, other_edges);

    if(DEBUG) cout << "Total Distance removed: " << distance_removed - distance_added << endl;

    if( distance_removed - distance_added > 0 ){
        if(DEBUG) cout << "Moving node " << *high_pos << " to just before " << *insert_pos << endl;

        int id = *high_pos;
        int prev_id = *prev(high_pos);
        int next_id = *next(high_pos);

        path.erase(high_pos);
        path.insert(insert_pos, id);

        edges.erase(edge(prev_id, id));
        edges.erase(edge(id, next_id));
        edges.insert(edge(prev_id, next_id));

    }
}

int find_worst_pos(list<int> &path, list<int>::iterator &high_pos, const set<edge> &used_edges, set<int> &refactored_ids){

    if(DEBUG) cout << "Fiding worst position...\n";
    int id, prev_id, next_id;
    int prev_distance, after_distance, high_distance;

    list<int>::iterator pos = path.begin();
    bool first = true;
    while( ++pos != path.end() && next(pos) != path.end() ){
        id = *pos;
        prev_id = *prev(pos);
        next_id = *next(pos);

        if(refactored_ids.find(id) != refactored_ids.end()){
            continue;
        }

        //if the NEW path created by REMOVING this node is already used
        if(used_edges.find(edge(prev_id, next_id)) != used_edges.end()){
            continue;
        }
        
        //this calculation is a bit painful, since in most cases it could be reused from the last iteration
        //however when a path is invalid, and an iteration is skipped, that would fail
        //it may be more effecient to always compute distances before checking paths
        //, and always use the previous value.
        prev_distance = sqrt(
            (float)((x[prev_id]-x[id])*(x[prev_id]-x[id]) + (y[prev_id]-y[id])*(y[prev_id]-y[id]))
        );

        after_distance = sqrt(
            (float)((x[next_id]-x[id])*(x[next_id]-x[id]) + (y[next_id]-y[id])*(y[next_id]-y[id]))
        );

        if(DEBUG2) cout << "Distance for " << prev_id << ":" << id << ":" << next_id << " " << prev_distance + after_distance << endl;
        if( first ){
            high_pos = pos;
            high_distance = prev_distance + after_distance;
        } else {
            if( (prev_distance + after_distance) > high_distance ){
                high_distance = prev_distance + after_distance;
                high_pos = pos;
            }
        }
    }
    if(DEBUG) cout << "Worst seq: " << *prev(high_pos) << ":"<< *high_pos << ":" << *next(high_pos) << endl;
    refactored_ids.insert(*high_pos);

    //these could be tracked along with high_pos, this seemed DRYer
    int high_prev = *prev(high_pos);
    int high_next = *next(high_pos);
    int removed_distance = sqrt(
        (float)((x[high_next]-x[high_prev])*(x[high_next]-x[high_prev]) + (y[high_next]-y[high_prev])*(y[high_next]-y[high_prev]))
    );
    if(DEBUG) cout << "Removed Distance: " << removed_distance << endl;
    int delta_removed = high_distance - removed_distance;
    if(DEBUG) cout << "Removed Distanced Delta: " << delta_removed << endl;


    return delta_removed;
}

int find_best_insert(const list<int>::iterator &high_pos, list<int> &path, list<int>::iterator &insert_pos, const set<edge> &used_paths) {

    list<int>::iterator pos;
    int low_distance, prev_distance, after_distance, id, prev_id;
    int high_id = *high_pos;
    if(DEBUG2) cout << "Finding best insert position for: " << high_id << endl;

    pos = path.begin();

    insert_pos = path.end();
    bool first = true;
    while(++pos != path.end()){
        prev_id = *prev(pos);
        id = *pos;

        //we can't consider where the old node was before being removed
        if(id == high_id || prev_id == high_id ){ 
            continue; 
        }

        //if either of the new paths after inserting are on the other path
        if( used_paths.find(edge(prev_id, high_id)) != used_paths.end() ||
            used_paths.find(edge(id, high_id)) != used_paths.end() ) {
            continue;
        }

        //find new distances inserting the id
        prev_distance = sqrt(
            (float)((x[prev_id]-x[high_id])*(x[prev_id]-x[high_id]) + (y[prev_id]-y[high_id])*(y[prev_id]-y[high_id]))
        );
        after_distance = sqrt(
            (float)((x[id]-x[high_id])*(x[id]-x[high_id]) + (y[id]-y[high_id])*(y[id]-y[high_id]))
        );

        if(DEBUG2) cout << "Distance for " << prev_id << ":" << high_id << ":" << id << " " << prev_distance + after_distance << endl;
        if( first ){
            low_distance = prev_distance + after_distance;
            insert_pos = pos;
        } else {
            if( prev_distance + after_distance < low_distance ){
                low_distance = prev_distance + after_distance;
                insert_pos = pos;
            }
        }
    }
    if( insert_pos == path.end() ){
        cout << "Error: Could not find valid insert point." << endl;
        exit(1);
    }
    if(DEBUG) cout << "Best prev: " << *prev(insert_pos) << ":" << *insert_pos << endl;

    id = *insert_pos;
    int insert_id_prev = *prev(insert_pos);
    int before_distance = sqrt(
        (float)((x[insert_id_prev]-x[id])*(x[insert_id_prev]-x[id]) + 
        (y[insert_id_prev]-y[id])*(y[insert_id_prev]-y[id]))
    );
    if(DEBUG) cout << "Before insert distance: " << before_distance << endl;
    int distance_added = low_distance - before_distance;
    if(DEBUG) cout << "Delta added: " << distance_added << endl;

    return distance_added;
}

/*
__global__ void compute_distances(int start, int *x, int *y, int *dist){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if( tid < NUM_NODES ){
        dist[tid] = sqrtf(
            (float)((x[tid]-x[start])*(x[tid]-x[start]) + (y[tid]-y[start])*(y[tid]-y[start]))
        );
    }
}
*/

