#include <iostream>
#include <stdio.h>
#include <vector>
#include <map>
#include <stdlib.h>
#include <queue>
using namespace std;

vector<vector<int>> crossover(int* par1, int* par2, int n, int left, int right) {
    vector<vector<int>> res;
    for (int tmp=0;tmp<2;tmp++){
        vector<int> a(n);
        res.push_back(a);
    }
    
    int count =2;
    int * p1 = par1;
    int * p2 = par2;
    while (count--){
        map<int, int> myMap;
        for (int i=left;i<=right;i++){
            res[1-count][i] = p1[i];
            myMap[p1[i]]=1;
        }
        int index = 0;
        for (int i=0;i<n;i++){
            // if item in parent2 not in range, add it to res
            if (myMap.find(p2[i]) == myMap.end()){
                while (index>=left && index <=right){
                    index++;
                }
                if (index >= n){
                    break;
                }
                res[1-count][index] = p2[i];
                index++;
            }
        }
        
        // exchange parent 1 and 2
        int* tmp = p1;
        p1 = p2;
        p2 = tmp;
    }
    return res;
}

int main() {
    int n;
    while (scanf("%d", &n)!=EOF){
        int par1[n];
        int par2[n];
        int left, right;
        for (int i = 0; i < n; i++)
            scanf("%d", &par1[i]);
        for (int i = 0; i < n; i++)
            scanf("%d", &par2[i]);
        
        scanf("%d", &left);
        scanf("%d", &right);

        auto res = crossover(par1, par2, n, left, right);
        for (int i=0;i<2;i++){
            for(int j=0;j<n;j++){
                printf("%d", res[i][j]);
                if (j!=n-1) {
                    printf(" ");
                }
            }
            printf("\n");
        }
    }
}