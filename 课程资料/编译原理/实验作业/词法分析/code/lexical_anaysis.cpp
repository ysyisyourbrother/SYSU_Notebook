#include "lexical_analysis.h"

// 运行词法分析代码
void LexicalAnalysis::run(ErrorMsgs& error_msgs, Tokens& tokens) {
	string filePath = "test_file.txt";

	ifstream file(filePath);

	if (!file.good()) {
		cout << "文件打开失败！" << endl;
		return;
	}

	cout << "词法分析结果为：" << endl;

	error_msgs = this->AnalyseTokens(file, tokens);

	for (auto token : tokens)
		cout << "(" << token.GetKindName() << " " << token.GetValue() << ")" << endl;

	for (auto msg : error_msgs)
		msg.print();
}

// 判断是否为特殊符号
bool LexicalAnalysis::is_special_symbol(char c) {
	return c == ':' || c == ',' || c == ';' ||
		c == '<' || c == '>' || c == '=' ||
		c == '+' || c == '-' || c == '*' ||
		c == '/' || c == '(' || c == ')' ||
		c == '{' || c == '}';
}

// 判断是否为单引号
bool LexicalAnalysis::is_single_quote(char c) {
	return c == '\'';
}

// 判断是否为空格
bool LexicalAnalysis::is_blank(char c) {
	return c == 0x20 || c == 0x09 || c == 0x0B || c == 0x0C || c == 0x0D;
}


// 判断是否为数字
bool LexicalAnalysis::is_number(char c) {
	return c >= '0' && c <= '9';
}

// 判断是否为字母
bool LexicalAnalysis::is_letter(char c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// 判断是否为新行
bool LexicalAnalysis::is_newline(char c) {
	return c == 0x0A;
}

// 获取当前字符的类型 如果都不是则认为是非法字符
LexicalAnalysis::CharType LexicalAnalysis::get_char_type(char c) {
	if (is_newline(c))
		return CharType::c_newline;
	if (is_number(c))
		return CharType::c_number;
	if (is_blank(c))
		return CharType::c_blank;
	if (is_special_symbol(c))
		return CharType::c_special_char;
	if (is_single_quote(c))
		return CharType::c_single_quote;
	if (is_letter(c))
		return CharType::c_letter;
	return CharType::c_illegal;
}

// 遍历文件生成Token
vector<ErrorMsg> LexicalAnalysis::AnalyseTokens(ifstream& sourceCode, Tokens& tokens) {
	// 定义当前扫描的行列
	int currLine = 0;
	unsigned int currColumn = 0;

	// 收集错误信息
	vector<ErrorMsg> errorMsgs;	

	// 定义初始状态为空白
	Status status = Status::blank;
	Tokens currLineTokens;
	CharType c_type;	// 字符的类型
	char c;	// 源代码中的字符
	string word;	// 源代码中的整个单词
	while (sourceCode.get(c)) {
		currColumn++;
		c_type = get_char_type(c);
		// 根据当前的状态，以及下一个字符来进行状态转移
		switch (status) {
		// 当前状态为空白
			case Status::blank:
				switch (c_type) {
				// 字符为换行 新开一行
				case CharType::c_newline:
					currLine++;
					currColumn = 0;
					tokens.append(currLineTokens);	// 记录当前生成的token
					currLineTokens.clear();	// 清空当前行的token内容
					break;
				// 字符为空格 直接跳过
				case CharType::c_blank:
					break;
				// 数字
				case CharType::c_number:
					word += c;
					status = Status::number;
					break;
				// 字母
				case CharType::c_letter:
					word += c;
					status = Status::letter;
					break;
				// 单引号
				case CharType::c_single_quote:
					status = Status::single_quote;
					break;
				// 特殊字符
				case CharType::c_special_char:
					if (c == '{') {	// 如果是左括号，进入注释模式
						status = Status::comment;
						break;
					}
					word = c;
					status = Status::special_char;	// 其他特殊符号，进入特殊符号模式
					break;
				// 非法字符
				case CharType::c_illegal:
					status = Status::error;
					errorMsgs.emplace_back(ErrorMsg(currLine, currColumn, "非法字符"));
					break;
				}
				break;
			// 当前状态为数字
			case Status::number:
				switch (c_type) {
				case CharType::c_newline:
					currLine++;
					currColumn = 0;
					currLineTokens.push({ tokenKind::NUM, word, currLine, currColumn - word.length()+1 });
					tokens.append(currLineTokens);
					currLineTokens.clear();
					word = "";
					status = Status::blank;
					break;
				case CharType::c_blank:
					currLineTokens.push({ tokenKind::NUM, word, currLine, currColumn - word.length()+1 });
					word = "";
					status = Status::blank;
					break;
				case CharType::c_number:
					word += c;
					break;
				// 进入数字状态后不能在转入字母状态，为非法
				case CharType::c_letter:
					errorMsgs.emplace_back(ErrorMsg(currLine, currColumn - word.length()+1, "数字和字母是非法组合"));
					word = "";
					status = Status::error;
					break;
				case CharType::c_single_quote:
					currLineTokens.push({ tokenKind::NUM, word, currLine, currColumn - word.length()+1 });
					word = "";
					status = Status::single_quote;
					break;
				case CharType::c_special_char:
					currLineTokens.push({ tokenKind::NUM, word, currLine, currColumn - word.length()+1 });
					word = "";
					if (c == '{') {
						status = Status::comment;
						break;
					}
					word = c;
					status = Status::special_char;
					break;
				case CharType::c_illegal:
					currLineTokens.push({ tokenKind::NUM, word, currLine, currColumn - word.length()+1 });
					word = "";
					status = Status::error;
					break;
				}
				break;
			// 当前状态为字母
			case Status::letter:
				switch (c_type) {
				case CharType::c_newline:
					if (Token::is_KEY(word))
						currLineTokens.push({ tokenKind::KEY, word, currLine, currColumn - word.length()+1 });
					else
						currLineTokens.push({ tokenKind::ID, word, currLine, currColumn - word.length()+1 });
					word = "";
					currLine++;
					currColumn = 0;
					tokens.append(currLineTokens);
					currLineTokens.clear();
					status = Status::blank;
					break;
				case CharType::c_blank:
					// 遇到空格，判断当前单词是否是关键字还是标识符
					if (Token::is_KEY(word))
						currLineTokens.push({ tokenKind::KEY, word, currLine, currColumn - word.length()+1 });
					else
						currLineTokens.push({ tokenKind::ID, word, currLine, currColumn - word.length()+1 });
					word = "";
					status = Status::blank;
					break;
				// 如果是字母的状态，遇到字母或者数字都维持字母状态
				case CharType::c_number:
				case CharType::c_letter:
					word += c;
					break;
				case CharType::c_single_quote:
					if (Token::is_KEY(word))
						currLineTokens.push({ tokenKind::KEY, word, currLine, currColumn - word.length()+1 });
					else
						currLineTokens.push({ tokenKind::ID, word, currLine, currColumn - word.length()+1 });
					word = "";
					status = Status::single_quote;
					break;
				case CharType::c_special_char:
					if (Token::is_KEY(word))
						currLineTokens.push({ tokenKind::KEY, word, currLine, currColumn - word.length()+1 });
					else
						currLineTokens.push({ tokenKind::ID, word, currLine, currColumn - word.length()+1 });
					word = c;
					status = Status::special_char;
					break;
				case CharType::c_illegal:
					if (Token::is_KEY(word))
						currLineTokens.push({ tokenKind::KEY, word, currLine, currColumn - word.length()+1 });
					else
						currLineTokens.push({ tokenKind::ID, word, currLine, currColumn - word.length()+1 });
					word = "";
					status = Status::error;
					errorMsgs.emplace_back(ErrorMsg(currLine, currColumn, "字母后面出现不合法的字符串"));
					break;
				}
				break;
			// 进入单引号状态，一直找到字符串结束的位置
			case Status::single_quote:
				switch (c_type) {
				// 单引号模式下不能换行
				case CharType::c_newline:
					errorMsgs.emplace_back(ErrorMsg(currLine, currColumn - word.length()+1, "字符串标识符'未匹配"));
					currLine++;
					currColumn = 0;
					currLineTokens.clear();
					status = Status::blank;
					break;
				// 向后一直遍历并保持当前状态，直到找到下一个单引号才停止
				case CharType::c_blank:
				case CharType::c_number:
				case CharType::c_letter:
				case CharType::c_special_char:
				case CharType::c_illegal:
					word += c;
					break;
				case CharType::c_single_quote:
					currLineTokens.push({ tokenKind::STR, word, currLine, currColumn - word.length()+1 });
					word = "";
					status = Status::blank;
					break;
				}
				break;
			// 当前状态为特殊符号
			case Status::special_char:
				// 如果不是注释状态 遇到右括号说明左括号缺失
				if (word == "}") {
					status = Status::error;
					errorMsgs.emplace_back(ErrorMsg(currLine, currColumn - word.length()+1, "注释标识符缺少匹配"));
					word = "";
					currLineTokens.clear();
					break;
				}
				switch (c_type) {
					case CharType::c_newline:
						currLine++;
						currColumn = 0;
						currLineTokens.push({ tokenKind::SYM, word, currLine, currColumn - word.length()+1 });
						tokens.append(currLineTokens);
						currLineTokens.clear();
						word = "";
						status = Status::blank;
						break;
					case CharType::c_blank:
						currLineTokens.push({ tokenKind::SYM, word, currLine, currColumn - word.length()+1 });
						word = "";
						status = Status::blank;
						break;
					case CharType::c_number:
						currLineTokens.push({ tokenKind::SYM, word, currLine, currColumn - word.length()+1 });
						word = c;
						status = Status::number;
						break;
					case CharType::c_letter:
						currLineTokens.push({ tokenKind::SYM, word, currLine, currColumn - word.length()+1 });
						word = c;
						status = Status::letter;
						break;
					case CharType::c_single_quote:
						currLineTokens.push({ tokenKind::SYM, word, currLine, currColumn - word.length()+1 });
						word = "";
						status = Status::single_quote;
						break;
					// 如果遇到连续的特殊字符，只有可能是>= <= 和:= 判断是否为这三种情况，如果不是出错
					case CharType::c_special_char:
						if ((word == ":" || word == "<" || word == ">") && c == '=') {
							word += c;
							currLineTokens.push({ tokenKind::SYM, word, currLine, currColumn - word.length()+1 });
							word = "";
							status = Status::blank;
						}
						else {
							word = "";
							status = Status::error;
							errorMsgs.emplace_back(ErrorMsg(currLine, currColumn, "非>=,<=,:=的连续特殊字符"));
						}
						break;
					case CharType::c_illegal:
						currLineTokens.push({ tokenKind::SYM, word, currLine, currColumn - word.length()+1 });
						status = Status::error;
						errorMsgs.emplace_back(ErrorMsg(currLine, currColumn, "特殊字符后面出现非法字符"));
						break;
					}
				break;
			// 当前状态为非法
			case Status::error:
				switch (c_type) {
				case CharType::c_newline:
					currLine++;
					currColumn = 0;
					status = Status::blank;
					break;
				case CharType::c_blank:
					status = Status::blank;
					break;
				case CharType::c_number:
				case CharType::c_letter:
				case CharType::c_single_quote:
				case CharType::c_special_char:
				case CharType::c_illegal:
					break;
				}
				break;
			case Status::comment:
				word += c;
				if (c == '}') {
					word = "";
					status = Status::blank;
				}
				if (is_newline(c)) {
					currLine++;
					currColumn = 0;
				}
				break;
		}
	}

	// 处理遍历结束后最后的字符
	switch (status) {
		case Status::blank:
			break;
		case Status::error:
		case Status::letter:
			if (Token::is_KEY(word))
				currLineTokens.push({ tokenKind::KEY, word, currLine, currColumn - word.length()+1 });
			else
				currLineTokens.push({ tokenKind::ID, word, currLine, currColumn - word.length()+1 });
			break;
		case Status::special_char:
			currLineTokens.push({ tokenKind::SYM, word, currLine, currColumn - word.length()+1 });
			break;
		case Status::single_quote:
			errorMsgs.emplace_back(ErrorMsg(currLine, currColumn - word.length()+1, "字符串缺少单引号匹配"));
			break;
		case Status::comment:
			errorMsgs.emplace_back(ErrorMsg(currLine, currColumn - word.length()+1, "注释标识符缺少匹配"));
			break;
		case Status::number:
			currLineTokens.push({ tokenKind::NUM, word, currLine, currColumn - word.length()+1 });
			break;
		}
	tokens.append(currLineTokens);
	return errorMsgs;
}




