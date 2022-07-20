#include <cstdio>
#include <cmath>
#include <vector>
 
using namespace std;
 
//存储城市之间的距离
vector<vector<double>> g;
//保存顶点i到状态s最后到达目标点的最小距离
vector<vector<double>> dp;
 
//核心函数，求出动态规划dp数组
void TSP(int N, int M){
    int end = N-1;   // 目标点
    //初始化dp[i][0]
    for(int i = 0 ; i < N ;i++){
        dp[i][0] = g[i][end];
    }
    //求解dp[i][j],先更新列在更新行
    for(int j = 1 ; j < M ;j++){
        for(int i = 0 ; i < N ;i++){
            dp[i][j] = -1;
            //如果中间点集合j中包含起始点i,则不符合条件退出
            if( ((j >> (i-1)) & 1) == 1){
                continue;
            }
            for(int k = 1 ; k < N ; k++){
                if( ((j >> (k-1)) & 1) == 0){
                    continue;
                }
                if ((dp[i][j] == -1) || (dp[i][j] > g[i][k] + dp[k][j^(1<<(k-1))])) {
                    dp[i][j] = g[i][k] + dp[k][j^(1<<(k-1))];
                }
            }
        }
    }
}
 
int main()
{
    int T;
    scanf("%d", &T);
    while (T--) {
        g.clear();
        dp.clear();
        int N;
        int x, y;
        scanf("%d", &N);
        int M = 1 << (N-1);
        // 初始化g和dp vector
        for (int tmp = 0;tmp<N;tmp++){
            vector<double> a(N);
            g.push_back(a);
        }
        for (int tmp = 0;tmp<N;tmp++){
            vector<double> b(M);
            dp.push_back(b);
        }
        
        vector<int> x_list;
        vector<int> y_list;
        for (int i=0;i<N;i++){
            scanf("%d %d", &x, &y);
            for (int j=0;j<x_list.size();j++){
                double dis = sqrt(pow(x-x_list[j],2) + pow(y-y_list[j],2));
                g[i][j] = dis;
                g[j][i] = dis;
            }
            x_list.push_back(x);
            y_list.push_back(y);
        }
        TSP(N,M);
        printf("%.2f\n",dp[0][M-1]);
    }
    return 0;
}