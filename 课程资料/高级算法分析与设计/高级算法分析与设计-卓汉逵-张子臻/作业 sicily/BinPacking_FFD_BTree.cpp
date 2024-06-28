#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <algorithm>
#include <queue>
 
using namespace std;

struct TreeNode {
    int capacity;   // 剩余容量
    TreeNode *left; // 左边是小于等于剩余容量的
    TreeNode *right; // 右边是大于剩余容量

    TreeNode(int c):capacity(c){}
};

// 层序遍历打印二叉树
void* levelOrder(TreeNode* root) {
    queue<TreeNode*> que;
    if(root != NULL) que.push(root);

    /* 开始层序遍历 */
    while(!que.empty()) {
        int size = que.size();
        for(int i = 0; i < size; i++) {
            TreeNode* node = que.front();
            que.pop();

            printf("%d ", node->capacity);
            if(node->left != NULL) que.push(node->left);
            if(node->right != NULL) que.push(node->right);

        }
    }
}

// 删除节点并重新插入
void* deleteNode(TreeNode* target, TreeNode* pre, int from) {
    TreeNode* newHead = NULL;
    if (target->left == NULL) {
        newHead = target->right;
    } else if (target->right == NULL) {
        newHead = target->left;
    } else {
        // 寻找右子树的最左节点
        TreeNode* node = target->right;
        while (node->left) {
            node = node->left;
        }
        node->left = target->left;
        newHead = target->right;
    }
    
    // 和父节点连接
    if (from == 0){
        pre->left = newHead;
    } else {
        pre->right = newHead;
    }
}

// 搜索一棵二叉树找到能容纳下的size的最大bin 找到则放入并返回0 找不到则返回-1
// prePoint 0 代表左 1 代表右
int place(TreeNode *root, int size, TreeNode *pre, int from){
    if (root == NULL) {
        return -1;
    }
    // 搜索第一个大于size的TreeNode
    while (root->capacity < size) {
        pre = root;
        from = 1;
        root = root->right;
        if (root == NULL) {
            return -1;
        }
    } 
    // 递归搜索
    int ret = place(root->left, size, root, 0);
    if (ret == -1) { // 后面没有更小的大于size的箱子，在当前bin插入
        ret = root->capacity - size;
        // 删除这个箱子并返回剩余capacity
        deleteNode(root, pre, from);
    }
    return ret;
}


// 插入一个箱子 
TreeNode* insertTree(TreeNode* root, int capacity) {
    // 创建新树
    if (root == NULL) {
        root = new TreeNode(capacity);
        return root;
    }
    // 递归插入
    TreeNode* cur = root;
    while (true) {
        if (capacity > cur->capacity) {
            if (cur->right == NULL) {
                cur->right = new TreeNode(capacity);
                break;
            } else {
                cur = cur->right;
            }
        } else {
            if (cur->left == NULL) {
                cur->left = new TreeNode(capacity);
                break;
            } else {
                cur = cur->left;
            }
        }
    }
    return root;
}

int binPackingTree(int *a, int size, int n)
{
    TreeNode *root = NULL;
    int ret;
    int binCount = 0;   // 统计创建了多少个bin
 
    for (int i = 0; i < n; i++){
        TreeNode* dummy = new TreeNode(-1);
        dummy->right = root;
        ret = place(root, a[i], dummy, 1);
        root = dummy->right;
        if (ret == -1) {  // 放置失败 创建新节点
            // printf("fail, 添加新节点：%d\n", a[i]);
            root = insertTree(root, size - a[i]);
            binCount ++;
        } else{
            // 放置成功，重新插入新箱子
            // printf("succ, 放入item：%d\n", a[i]);
            root = insertTree(root, ret);
        }
        // printf("遍历二叉树：");
        // levelOrder(root);
        // printf("\n");
    }

    return binCount;
}
 
int binPacking(int *a, int size, int n) {
    TreeNode *root;
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
        if (size == 0) {
            printf("0\n");
            continue;
        }
        sort(a, a+n, greater<int>());
        int out = binPackingTree(a, size, n);
        printf("%d\n", out);
    }
}