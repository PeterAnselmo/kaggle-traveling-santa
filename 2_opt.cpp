//#include <bitset>
#include <algorithm>
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
void make_2_opt(vector<int>&, set<edge>&, set<edge>&);

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
    //convert to vectors for random access
    vector<int> path1_vector;
    vector<int> path2_vector;
    copy(path1.begin(), path1.end(), back_inserter(path1_vector));
    copy(path2.begin(), path2.end(), back_inserter(path2_vector));

    make_2_opt(path1_vector, used_edges1, used_edges2);
    make_2_opt(path2_vector, used_edges2, used_edges1);

    cout << "path1\tpath2" << endl;
    for(int i=0; i<num_nodes; ++i){
        cout << path1_vector[i] << "\t" << path2_vector[i] << endl;
    }

}

void make_2_opt(vector<int> &path, set<edge> &used_edges, set<edge> &other_edges){

    int a, b, c, d;
    for(int i=3; i<num_nodes; ++i){
        a = path[i-3];
        b = path[i-2];
        c = path[i-1];
        d = path[i];

        //check if swapping would be valid before computing paths
        if( other_edges.find(edge(a,c)) != other_edges.end() ||
            other_edges.find(edge(b,d)) != other_edges.end() ){
            continue;
        }

        if(
            (sqrt(
                (float)((x[a]-x[c])*(x[a]-x[c]) + (y[a]-y[c])*(y[a]-y[c]))
            ) +
            sqrt(
                (float)((x[b]-x[d])*(x[b]-x[d]) + (y[b]-y[d])*(y[b]-y[d]))
            )) 
            <
            (sqrt(
                (float)((x[a]-x[b])*(x[a]-x[b]) + (y[a]-y[b])*(y[a]-y[b]))
            ) +
            sqrt(
                (float)((x[c]-x[d])*(x[c]-x[d]) + (y[c]-y[d])*(y[c]-y[d]))
            )) 
          ){
            path[i-2] = c;
            path[i-1] = b;

            used_edges.erase(edge(a,b));
            used_edges.erase(edge(c,d));
            used_edges.insert(edge(a,c));
            used_edges.insert(edge(b,d));
        }
    }

}
