#include "datatype.h"


// 定义所有token种类的名字
vector<string> Token::kind_names_ = {
		"KEY",		//保留字
		"SYM",		//特殊符号
		"ID",		//标识符
		"NUM",		//数值常数
		"STR",		//字符串常数
		"TK_TRUE",
		"TK_FALSE",
		"TK_OR",
		"TK_AND",
		"TK_NOT",
		"TK_INT",
		"TK_BOOL",
		"TK_STRING",
		"TK_WHILE",
		"TK_DO",
		"TK_IF",
		"TK_THEN",
		"TK_ELSE",
		"TK_END",
		"TK_REPEAT",
		"TK_UNTIL",
		"TK_READ",
		"TK_WRITE",
		"TK_GTR",
		"TK_LEQ",
		"TK_GEQ",
		"TK_COMMA",
		"TK_SEMICOLON",
		"TK_ASSIGN",
		"TK_ADD",
		"TK_SUB",
		"TK_MUL",
		"TK_DIV",
		"TK_LP",
		"TK_RP",
		"TK_LSS",
		"TK_EQU"
};

// 定义默认构造函数
Token::Token() {}

// 定义构造函数
Token::Token(tokenKind kind_, std::string value_, int line_, unsigned int column_) {
	kind = kind_;
	value = value_;
	line = line_;
	column = column_;
}

// 获取当前token类型，是一个整数类型，需要用GetKindName获取名字
tokenKind Token::GetKind() {
	return this->kind;
}

// 获取当前单词串的内容
string Token::GetValue() {
	return this->value;
}

// 获取当前行数
int Token::GetLine() {
	return this->line;
}

// 获取当前列数
unsigned long Token::GetColumn() {
	return this->column;
}

// 返回token类别名 因为是enum类型需要做一个转换
string Token::GetKindName() {
	return this->kind_names_[static_cast<int>(this->kind)];
}


string Token::GetKindName(tokenKind kind) {
	return kind_names_[static_cast<int>(kind)];
}

// 定义语法中所有可能出现的关键词
bool Token::is_KEY(string& str) {
	return str == "true" || str == "false" || str == "or" ||
		str == "and" || str == "not" || str == "int" ||
		str == "bool" || str == "string" || str == "while" ||
		str == "do" || str == "if" || str == "then" ||
		str == "else" || str == "end" || str == "repeat"
		|| str == "until" || str == "read" || str == "write";
}


// 实现token的方法
void Tokens::push(Token token) {
	tokens_.emplace_back(token);
}


void Tokens::push(tokenKind kind, std::string value, int line, unsigned long column) {
	Token token(kind, value, line, column);
	tokens_.emplace_back(token);
}


void Tokens::clear() {
	tokens_.clear();
}


void Tokens::append(Tokens tokens) {
	tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
}


vector<Token>::iterator Tokens::begin() {
	return tokens_.begin();
}


vector<Token>::iterator Tokens::end() {
	return tokens_.end();
}


Token Tokens::back() {
	return tokens_.back();
}


unsigned long Tokens::size() const {
	return tokens_.size();
}


// 实现Error方法
std::string ErrorMsg::to_string() {
	return "在第" + std::to_string(line) + "行, 第" + std::to_string(column) + "列出错: " + msg;
}


void ErrorMsg::print() {
	std::cout << to_string() << std::endl;
}

