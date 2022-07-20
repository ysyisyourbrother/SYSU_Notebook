#ifndef GRAMMAR_ANALYSIS_H
#define GRAMMAR_ANALYSIS_H


#include "datatype.h"


class GrammarAnalysis {
public:
	TreeNode* root = NULL;

	Tokens tokens_;
	std::vector<Token>::iterator token_it_;
	using KindSet = std::set<tokenKind>;

	GrammarAnalysis(Tokens token);

	// 判断当前的token（即当前迭代器）的kind值是否属于kind_set
	bool match(KindSet kind_set, std::string eof_error_msg = "") throw(ErrorMsg);

	// program是开始符，对应的分析函数的返回值就是完整的语法树的root
	// 该分析函数主要就调用declaration和stmt_sequence
	TreeNode* program(ErrorMsgs& error_msg);

	// 分析变量的声明
	void declaration();

	// 分析语句列表，返回值就是整个源代码的语法树（不包括声明部分）
	TreeNode* stmt_sequence();

	// if-stmt产生式的分析函数
	TreeNode* if_stmt();
	// 下面的分析函数同理，是对应的产生式的分析函数
	TreeNode* repeat_stmt();
	TreeNode* assign_stmt();
	TreeNode* read_stmt();
	TreeNode* write_stmt();
	TreeNode* while_stmt();
	TreeNode* log_or_exp();
	TreeNode* log_and_exp();
	TreeNode* comparision_exp();
	TreeNode* add_exp();
	TreeNode* mul_exp();
	TreeNode* factor();

	// 层序遍历打印语法树
	void print_tree();
};

#endif
