#include <stdio.h>
#include <vector>
using namespace std;

struct Node{
    // objective function values
    int x;
    int y;

    int rank;   // front level
    int n;     // number of node dominate it
    vector<int> Set;    // nodes who dominated by it
};

// whether x1 dominate x2 
bool dominate(int x1, int x2, vector<Node> &nodes){
    if ((nodes[x1].x <= nodes[x2].x) && (nodes[x1].y <= nodes[x2].y)){
        if((nodes[x1].x < nodes[x2].x) || (nodes[x1].y < nodes[x2].y)){
            return true; 
        } 
    }
    return false;
}

int main() {
    int N;
    scanf("%d", &N);
    vector<Node> nodes;
    for(int i=0;i<N; i++){
        Node node;
        scanf("%d %d", &node.x, &node.y);
        nodes.push_back(node);
    }
    vector<int> Fi;
    for(int p=0;p<N;p++){
        nodes[p].n = 0;
        for(int q=0;q<N;q++){
            if(dominate(p, q, nodes)){
                nodes[p].Set.push_back(q);
            } 
            else if(dominate(q, p, nodes)){
                nodes[p].n++;
            }
        }
        if (nodes[p].n == 0){
            nodes[p].rank = 1;
            Fi.push_back(p);
        }
    }
    int level = 1;
    while (Fi.size() > 0){
        vector<int> Q;
        for (int i=0;i<Fi.size();i++){
            int p = Fi[i];
            for(int j=0;j<nodes[p].Set.size();j++){
                int q = nodes[p].Set[j];
                nodes[q].n--;
                if (nodes[q].n == 0) {
                    nodes[q].rank = level + 1;
                    Q.push_back(q);
                }
            }
        }
        level ++;
        Fi = Q;
    }
    for(int i=0;i<N;i++){
        printf("%d\n", nodes[i].rank);
    }
}