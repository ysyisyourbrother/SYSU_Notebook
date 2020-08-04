#include"lexical_analysis.h"
#include"grammar_analysis.h"

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
}

int main() {
	run_all();
	system("pause");
}