#include"lexical_analysis.h"

void run_all() {
	// 定义错误信息
	ErrorMsgs error_msgs;

	// 定义tokens
	Tokens tokens;

	// 运行词法分析结果
	LexicalAnalysis* lexicalAnalysis = new LexicalAnalysis();
	lexicalAnalysis->run(error_msgs, tokens);

}

int main() {
	run_all();
	system("pause");
}