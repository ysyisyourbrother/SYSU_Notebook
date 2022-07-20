#include "headfile.h"
using namespace std;

void recursion_E(vector<string>, int, int);
void recursion_T(vector<string>, int, int);
void recursion_F(vector<string>, int, int);
vector<string> res; // 最终结果


vector<string> split(string s) {
    // 提前将字符串分割成多个字串列表，方便递归操作

    vector<string> expre(100,"");
	bool flag = 0;
	int count = 0; // 收集所有分开后的数值或者符号
	for (int i=0; i<s.length(); i++) {
		// 判断是否是value，小数需要额外判断小数点
        // 这里没有合法性的判断
		if ((s[i] >= '0' && s[i] <= '9') || s[i] == '.'){
			expre[count] += s[i];
			flag = 1;
		}
		// 遇到符号停止
		else if (s[i] == '+' || s[i] == '-' || s[i] == '*' || 
                s[i] == '/' || s[i] == '(' || s[i] == ')'){
			if (flag) {
				count++;
				flag = 0;
			}
			// 专门处理负数
			if (s[i-1] == '(' && (s[i] == '+' || s[i] == '-')) {
				int j;
				for (j=i; j<s.length() && s[j] != ')'; ++j)
					expre[count] += s[j];
				i = j-1;
				count++;
			}
			else 
				expre[count++] = s[i];
		}
	}
	return expre;
}

void recursion_E(vector<string> expre, int start, int end) {
	int i;
	int count = 0;
	// 从后往前不在括号中的第一个加号或减号 
	for (i=end; i>=start; --i) {
		if (expre[i] == "(") count--;
		if (expre[i] == ")") count++;
		if (!count && (expre[i] == "+" || expre[i] == "-"))  // 如果找到，退出
            break;
	}
    // 对第一个加号或者减号左右两边的进行递归
	if (i >=start) {
        // 以这个符号为分界线将表达式分为左右两边
        // 类似二叉树的后序遍历方法，符号是根，加入res中
		recursion_E(expre, start, i-1); 
		recursion_T(expre, i+1, end);
        res.push_back(expre[i]);
	}
    // 如果整个表达式都被括号包围例如:(1+2) ，整个表达式一起进入T函数
	else 
        recursion_T(expre, start, end);
}

void recursion_T(vector<string> expre, int start, int end) {
    // 进入T函数后，要么是乘除法 要么就是F
	int i;
	int count = 0; // 记录括号层数
	// 第一个不在括号的乘号或除号
	for (i=end; i>=start; --i) {
		if (expre[i] == "(") count--;
		if (expre[i] == ")") count++;
		if (!count && (expre[i] == "*" || expre[i] == "/"))  // 如果找到，退出
            break;
	}
    // 对第一个乘号或者除号左右两边的进行递归
	if (i >= start) {
		recursion_T(expre, start, i-1);
		recursion_F(expre, i+1, end);
        res.push_back(expre[i]);
	}
    // 如果不存在括号外面的*/号，只能是F，整个进入F函数
	else 
        recursion_F(expre, start, end);
}

void recursion_F(vector<string> expre, int start, int end) {
    // 判断是否是一个value或者(E)
    // 如果是一个value直接加入res中，如果是一个(E)把括号拆掉继续判断
	if (expre[start] == "(")  // 如果开始还是一个括号，说明不是，还可以继续拆解，因此继续递归。
        recursion_E(expre, start+1, end-1);
	else 
        // 如果没有括号了，说明已经是一个value，直接加入答案队列中
        res.push_back(expre[start]);
}	


int main() {
	while (1) {
        // 初始化存储空间
        string s;
        res.clear();

        // 输入待转化的表达式
		cout << "请输入算术表达式（输入q结束）：" << endl;
		getline(cin, s);
		if (s == "q") break; 
		vector<string> expre = split(s); 

        // 递归调用，得到结果
        recursion_E(expre, 0, expre.size()-1);

        //输出转化后的表达式
		cout << "其逆波兰表达式为：" << endl; 
        for(auto i : res)
            cout<<i<<" ";

		cout << endl << endl;
	}
	return 0;
}

/*
3*(4+5/(2-1))
1.414 + 3.666 / (1.333-5.893)
21+42-30/(5+5)*(4-2)
*/
