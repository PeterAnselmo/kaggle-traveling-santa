#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct coord{
    int x, y;
};

void compute_random_path(vector<int> &path1, vector<int> &path2);
int shuffle_front(deque<int> &nodes);
long double compute_score(const vector<int> &path1, const vector<int> &path2);
long double compute_distance(const vector<int> &path);
void print_paths(const vector<int> &path1, const vector<int> path2, char* outname);

const int NUM_ITERATIONS = 300;
vector<coord> coords;
int num_nodes;

int main(int argc, char *argv[]){
    srand ( time(NULL) );

    ifstream fh("test.txt");

    string line;
    getline(fh, line);//header row
    int id;
    while(fh >> id){
        coord temp;
        fh >> temp.x;
        fh >> temp.y;
        //cout << temp.x << ":" << temp.y << endl;
        coords.push_back(temp);
    }
    num_nodes = coords.size();

    int num_elements = coords.size();
    vector<int> path;
    path.reserve(num_elements);

    vector<int> path1, path2, final_path1, final_path2;
    long double score, low_score;

    compute_random_path(path1, path2);
    low_score = compute_score(path1, path2);
    cout << "Initial Low Score: " << low_score << endl;
    print_paths(path1, path2, argv[1]);

    int count = 0;
    for(int i=0; i<NUM_ITERATIONS; ++i){
        compute_random_path(path1, path2);
        score = compute_score(path1, path2);
        cout << "Last Score: " << score << endl;
        if( score < low_score ){
            cout << "New Low!" << endl;
            low_score = score;
            final_path1 = path1;
            final_path2 = path2;
            print_paths(path1, path2, argv[1]);
        }

        if( ++count %10 == 0 ){
            cout << count << " iterations tried.\n";
        }
    }
    
}

void print_paths(const vector<int> &path1, const vector<int> path2, char* outname){

    ofstream fh(outname);

    int path1_size, path2_size;
    path1_size = path1.size();
    path2_size = path2.size();
    if( path1_size != path2_size ){
        cout << "ERROR, path size are not the same\n";
        exit(1);
    }

    fh << "path1\tpath2" << endl;
    for(int i=0; i<path1_size; ++i){
        fh << path1[i] << "\t" << path2[i] << endl;
    }
}

void compute_random_path(vector<int> &path1, vector<int> &path2){
    
    //will be a set of strings "1234-4312"
    set<string> used_edges1, used_edges2;
    deque<int> nodes1, nodes2;

    path1.clear();
    path2.clear();

    //prefill with consecutively numbered nodes
    for(int i=0; i<num_nodes; ++i){
        nodes1.push_back(i);
        nodes2.push_back(i);
    }

    /*
    random_shuffle(nodes1.begin(), nodes1.end());
    random_shuffle(nodes2.begin(), nodes2.end());
    */

    int pos1, last_pos1, pos2, last_pos2;
    last_pos1 = shuffle_front(nodes1);
    nodes1.pop_front();
    path1.push_back(last_pos1);

    last_pos2 = shuffle_front(nodes2);
    nodes2.pop_front();
    path2.push_back(last_pos2);

    //should execute n-1 times
    string key;
    int sentinel;
    int max_iterations = 1000;
    for(int i=1; i<num_nodes; ++i){

        sentinel = 0;
        do {
            if( ++sentinel > max_iterations ){
                cout << "Stuck on traveled paths" << endl;
                exit(1);
            }

            //get a candiate next node
            pos1 = shuffle_front(nodes1);

            //compute hash key with last node
            if( pos1 > last_pos1 ){
                key = to_string(last_pos1) + "-" + to_string(pos1);
            } else {
                key = to_string(pos1) + "-" + to_string(last_pos1);
            }

        //if candidate path was already used
        } while(used_edges2.find(key) != used_edges2.end());

        //front element will be random shuffled element just used
        nodes1.pop_front();
        path1.push_back(pos1);
        used_edges1.insert(key);
        last_pos1 = pos1;
        
        sentinel = 0;
        do {
            if( ++sentinel > max_iterations ){
                cout << "Stuck on traveled paths" << endl;
                exit(1);
            }

            //get a candiate next node
            pos2 = shuffle_front(nodes2);

            //compute hash key with last node
            if( pos2 > last_pos2 ){
                key = to_string(last_pos2) + "-" + to_string(pos2);
            } else {
                key = to_string(pos2) + "-" + to_string(last_pos2);
            }

        //if candidate path was already used
        } while(used_edges1.find(key) != used_edges1.end());

        //front element will be random shuffled element just used
        nodes2.pop_front();
        path2.push_back(pos2);
        used_edges2.insert(key);
        last_pos2 = pos2;
    }
}

int shuffle_front(deque<int> &nodes){
    int temp_idx = rand() % nodes.size();
    int temp = nodes[temp_idx];
    nodes[temp_idx] = nodes[0];
    nodes[0] = temp;

    return temp;
}

long double compute_score(const vector<int> &path1, const vector<int> &path2){
    long double distance1, distance2;

    distance1 = compute_distance(path1);
    distance2 = compute_distance(path2);

    return (distance1 > distance2) ? distance1 : distance2;
}

long double compute_distance(const vector<int> &path){
    long double distance = 0;

    int last_pos = path[0];
    int path_size = path.size();
    for(int i=1; i<path_size; ++i){
        int pos = path[i];
        distance += sqrt( 
           (coords[pos].x-coords[last_pos].x)*(coords[pos].x-coords[last_pos].x) + 
           (coords[pos].y-coords[last_pos].y)*(coords[pos].y-coords[last_pos].y)
       );
       last_pos = pos;

    }
    return distance;
}
