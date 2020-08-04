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

// 细化token类型
void Token::RefineKind() {
	if (kind == tokenKind::ID || kind == tokenKind::STR || kind == tokenKind::NUM)
		return;
	if (kind == tokenKind::KEY) {
		if (value == "true")
			kind = tokenKind::TK_TRUE;
		if (value == "false")
			kind = tokenKind::TK_FALSE;
		if (value == "or")
			kind = tokenKind::TK_OR;
		if (value == "and")
			kind = tokenKind::TK_AND;
		if (value == "not")
			kind = tokenKind::TK_NOT;
		if (value == "int")
			kind = tokenKind::TK_INT;
		if (value == "bool")
			kind = tokenKind::TK_BOOL;
		if (value == "string")
			kind = tokenKind::TK_STRING;
		if (value == "while")
			kind = tokenKind::TK_WHILE;
		if (value == "do")
			kind = tokenKind::TK_DO;
		if (value == "if")
			kind = tokenKind::TK_IF;
		if (value == "then")
			kind = tokenKind::TK_THEN;
		if (value == "else")
			kind = tokenKind::TK_ELSE;
		if (value == "end")
			kind = tokenKind::TK_END;
		if (value == "repeat")
			kind = tokenKind::TK_REPEAT;
		if (value == "until")
			kind = tokenKind::TK_UNTIL;
		if (value == "read")
			kind = tokenKind::TK_READ;
		if (value == "write")
			kind = tokenKind::TK_WRITE;
	}
	if (kind == tokenKind::SYM) {
		if (value == ">")
			kind = tokenKind::TK_GTR;
		if (value == "<=")
			kind = tokenKind::TK_LEQ;
		if (value == ">=")
			kind = tokenKind::TK_GEQ;
		if (value == ",")
			kind = tokenKind::TK_COMMA;
		if (value == ";")
			kind = tokenKind::TK_SEMICOLON;
		if (value == ":=")
			kind = tokenKind::TK_ASSIGN;
		if (value == "+")
			kind = tokenKind::TK_ADD;
		if (value == "-")
			kind = tokenKind::TK_SUB;
		if (value == "*")
			kind = tokenKind::TK_MUL;
		if (value == "/")
			kind = tokenKind::TK_DIV;
		if (value == "(")
			kind = tokenKind::TK_LP;
		if (value == ")")
			kind = tokenKind::TK_RP;
		if (value == "<")
			kind = tokenKind::TK_LSS;
		if (value == "=")
			kind = tokenKind::TK_EQU;
	}
}


// 实现tokens的方法
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

void Tokens::RefineKind() {
	for (auto& token : tokens_) {
		token.RefineKind();
	}
}


// 实现Error方法
std::string ErrorMsg::to_string() {
	return "在第" + std::to_string(line) + "行, 第" + std::to_string(column) + "列出错: " + msg;
}


void ErrorMsg::print() {
	std::cout << to_string() << std::endl;
}

vector<string> TreeNode::type_names = {
		"PROGRAM",		// 程序（开始符号）节点
		"STMT_SEQUENCE",// 语句列表节点
		"IF_STMT",      // 条件语句节点
		"REPEAT_STMT",  // repeat语句节点
		"ASSIGN_STMT",  // 赋值语句节点
		"READ_STMT",    // read语句节点
		"WRITE_STMT",   // write语句节点
		"WHILE_STMT",   // while语句节点
		"GTR_EXP",      // 大于表达式节点
		"GEQ_EXP",      // 大于等于表达式节点
		"LSS_EXP",      // 小于表达式节点
		"LEQ_EXP",      // 小于等于表达式节点
		"EQU_EXP",      // 等于表达式节点
		"LOG_OR_EXP",   // 逻辑或表达式节点
		"LOG_AND_EXP",  // 逻辑与表达式节点
		"LOG_NOT_EXP",  // 逻辑非表达式节点
		"ADD_EXP",      // 加法表达式节点
		"SUB_EXP",      // 减法表达式节点
		"MUL_EXP",      // 乘法表达式节点
		"DIV_EXP",      // 除法表达式节点
		"FACTOR"		// 原子节点
};


// 实现语法树类的方法
TreeNode::TreeNode(TreeNode::Type type, TreeNode* t1, TreeNode* t2, TreeNode* t3) {
	type_ = type;
	child_[0] = t1;
	child_[1] = t2;
	child_[2] = t3;
}


TreeNode::TreeNode(TreeNode::Type type, Token token) {
	type_ = type;
	tk_ = token;
}


string TreeNode::get_type_name() {
	return type_names[static_cast<int>(type_)];
}


string TreeNode::type_name(TreeNode::Type type) {
	return type_names[static_cast<int>(type)];
}


// 符号表实现
Sym* SymTable::insert(std::string name) {
	if (table_.find(name) != table_.end())
		return NULL;
	Sym* sym = new Sym();
	table_.insert({ name, sym });
	return sym;
}


Sym* SymTable::find(std::string name) {
	if (table_.find(name) != table_.end())
		return table_[name];
	return NULL;
}


void SymTable::del(std::string name) {
	auto it = table_.find(name);
	if (it != table_.end())
		table_.erase(it);
}


void SymTable::print() {
	for (auto it : table_) {
		cout << it.first << " ";
		if (it.second->val_type == ValType::VT_STRING)
			cout << "string类型" << endl;
		if (it.second->val_type == ValType::VT_INT)
			cout << "int类型   " << endl;
		if (it.second->val_type == ValType::VT_BOOL)
			cout << "bool类型  " << endl;
	}
}
