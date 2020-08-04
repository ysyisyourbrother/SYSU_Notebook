#include"lexical_analysis.h"
#include"grammar_analysis.h"
#include"intermediate_code.h"

void run_all() {
	// 定义错误信息
	ErrorMsgs error_msgs;

	// 定义tokens
	Tokens tokens;

	// 运行词法分析结果
	LexicalAnalysis* lexicalAnalysis = new LexicalAnalysis();
	lexicalAnalysis->run(error_msgs, tokens);

	// 进一步细分token类型
	tokens.RefineKind();

	// 得到语法分析结果
	GrammarAnalysis* grammarAnalysis = new GrammarAnalysis(tokens);
	grammarAnalysis->root = grammarAnalysis->program(error_msgs);
	grammarAnalysis->print_tree();

	// 生成符号表
	cout << endl << "符号表：" << endl;
	grammarAnalysis->print_sym_table();

	// 生成中间代码
	cout << endl << "中间代码生成结果：" << endl;
	IntermediateCode* intermediateCode = new IntermediateCode();
	if (grammarAnalysis->root != NULL){
		intermediateCode->stmt_seq_node(grammarAnalysis->root);
	}

	// 打印中间代码结果 label顶格，其他打印一个缩进
	vector<string> interCode = intermediateCode->res_;
	for (auto i : interCode) {
		if (i.length() > 5 && i.substr(0, 5) == "Label")cout << i << endl;
		else cout << "  " << i << endl;
	}
}

int main() {
	run_all();
	system("pause");
}