#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include "tsp.h"

using namespace std;

int main(int argc, char *argv[]){

    int num_nodes = get_num_nodes(argv[1]);

    //keep track of used paths to make sure every 
    //node is used exactly once
    bool *used1, *used2;
    used1 = new bool[num_nodes];
    used2 = new bool[num_nodes];

    for(int i=0; i<num_nodes; ++i){
        used1[i] = false;
        used2[i] = false;
    }

    //store the coordinate positions for lookups
    int x[num_nodes], y[num_nodes];

    read_coords(argv[1], num_nodes, x, y);

    ifstream solution(argv[2]);

    string line;
    getline(solution, line); //discard header row

    int pos1, last_pos1, pos2, last_pos2;
    double distance1, distance2;
    distance1 = distance2 = 0;
    solution >> last_pos1;
    solution >> last_pos2;
    used1[last_pos1] = true;
    used2[last_pos2] = true;

    while(solution >> pos1){
        solution >> pos2;
        used1[pos1] = true;
        used2[pos2] = true;
        distance1 += sqrt( 
           (x[pos1]-x[last_pos1])*(x[pos1]-x[last_pos1]) + 
           (y[pos1]-y[last_pos1])*(y[pos1]-y[last_pos1])
       );
        distance2 += sqrt( 
           (x[pos2]-x[last_pos2])*(x[pos2]-x[last_pos2]) + 
           (y[pos2]-y[last_pos2])*(y[pos2]-y[last_pos2])
       );
       last_pos1 = pos1;
       last_pos2 = pos2;
    }

    for(int i=0; i<num_nodes; ++i){
        if( !used1[i] ){
            cout << "Error: " << i << " missing from path 1" << endl;
        }
        if( !used2[i] ){
            cout << "Error: " << i << " missing from path 2" << endl;
        }
    }

    cout << distance1 << "," << distance2 << endl;

}
