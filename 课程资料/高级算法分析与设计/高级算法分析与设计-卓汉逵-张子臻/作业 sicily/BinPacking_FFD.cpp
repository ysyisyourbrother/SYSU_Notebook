#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <algorithm>
 
using namespace std;
 
int binPacking(int *a, int size, int n)
{
    int binCount = 0;
    int binValues[n];
    for (int i = 0; i < n; i++)
        binValues[i] = size;
 
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            if (binValues[j] - a[i] >= 0)
            {
                binValues[j] -= a[i];
                break;
            }
        }
 
    for (int i = 0; i < n; i++)
        if (binValues[i] != size)
            binCount++;
    
    return binCount;
}
 
int main()
{
    int n;
    int size;
    while (scanf("%d %d", &n, &size)!=EOF){
        int a[n];
        for (int i = 0; i < n; i++)
            scanf("%d", &a[i]);
        sort(a, a+n, greater<int>());
        int out = binPacking(a, size, n);
        printf("%d\n", out);
    }
}