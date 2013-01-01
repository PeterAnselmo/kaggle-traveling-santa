#ifndef TSP_H
#define TSP_H

#include <fstream>
#include <list>
#include <set>

const bool DEBUG = false;
const bool DEBUG2 = false; //more verbose debugging

using namespace std;

/* edge - a struct to represent a path between two nodes.
 * the main logic meriting a struct is the constructur that
 * always stores the path from low id to high id.  Since
 * the paths are bidirectional, this ensures the direction
 * of travel is not recorded.
 */
struct edge{
    int start, end;

    //paths are not directional, alway store lowest to highest
    edge(int id1, int id2){
        if( id1 < id2 ){
            start = id1;
            end = id2;
        } else {
            start = id2;
            end = id1;
        }
    }
    bool operator <(const edge &rhs) const {
        if( start == rhs.start ){
            return end < rhs.end;
        } else {
            return start < rhs.start;
        }
    }
    bool operator ==(const edge &rhs) const {
        return (start == rhs.start) && (end == rhs.end);
    }
    bool operator !=(const edge &rhs) const {
        return (start != rhs.start) || (end != rhs.end);
    }
};


/* read in coordinates from a file. Assumes 3 space seperated
 * columns: id, x, y. Also, assumes and discards a header row.
 */
void read_coords(char* filename, int num_nodes, int x[], int y[]){

    ifstream fh(filename);
    if( fh.fail() ){
        cout << "Error opening coordinate file." << endl;
        exit(1);
    }
    
    //discard header row
    string line;
    getline(fh, line);

    int id;
    while(fh >> id){
        fh >> x[id];
        fh >> y[id];
    }
    fh.close();

    if(DEBUG) cout << "Coords Read." << endl;
}

void read_paths(char* filename, 
        set<edge> &used_edges1,
        set<edge> &used_edges2,
        list<int> &path1,
        list<int> &path2){

    ifstream fh(filename);
    if( fh.fail() ){
        cout << "Error opening path file." << endl;
        exit(1);
    }

    //discard header row
    string line;
    getline(fh, line);

    int id1, id2, last_id1, last_id2;
    bool first = true;
    while((fh >> id1)){
        fh >> id2;
        if( !first ){
            used_edges1.insert(edge(last_id1, id1));
            used_edges2.insert(edge(last_id2, id2));
        }
        first = false;
        path1.push_back(id1);
        path2.push_back(id2);
        last_id1 = id1;
        last_id2 = id2;
    }
    fh.close();

    if(DEBUG) cout << "Paths Read." << endl;
}


//http://stackoverflow.com/questions/478898/how-to-execute-a-command-and-get-output-of-command-within-c
std::string exec(string cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "ERROR";
    char buffer[128];
    std::string result = "";
    while(!feof(pipe)) {
        if(fgets(buffer, 128, pipe) != NULL)
            result += buffer;
    }
    pclose(pipe);
    return result;
}

int get_num_nodes(char* filename){

    string wc_cmd = "wc -l ";
    wc_cmd += filename;

    int num_nodes;
    num_nodes = atoi(exec(wc_cmd).c_str()) - 1;
    if(DEBUG) cout << "Num nodes detected: " << num_nodes << endl;

    return num_nodes;
}

#endif
