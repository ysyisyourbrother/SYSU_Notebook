#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <queue>
using namespace std;

int binPackingTree(int *item_size, int q, int n) {
    vector<int> tree(2*n, q);
    for(int i=0; i<n; i++) {
        int node = 2;
        while(node < n) {
            if(item_size[i] > tree[node]) {
                node++;
            }
            
            if(node < n) {
                node *= 2;
            } 
        }
        if(item_size[i] > tree[node]) {
            node++;
        }
        tree[node] -= item_size[i];
        node /= 2;
        while(node > 0) {
            int max_size = max(tree[2 * node], tree[2 * node + 1]);
            if(max_size == tree[node]) {
                break;
            }
            tree[node] = max_size;
            node /= 2;
        }
    }
    int ans = 0;
    for(int p=tree.size()-1; p>=tree.size()-n; p--){
        if(tree[p] != q) {
            ans++;
        }
    } 
    return ans;
}

int main() {
    int n;
    int size;
    while (scanf("%d %d", &n, &size)!=EOF){
        int a[n];
        for (int i = 0; i < n; i++)
            scanf("%d", &a[i]);
        if (size == 0) {
            printf("0\n");
            continue;
        }
        sort(a, a+n, greater<int>());
        int out = binPackingTree(a, size, n);
        printf("%d\n", out);
    }
}