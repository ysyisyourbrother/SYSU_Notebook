#ifndef DATATYPE_H
#define DATATYPE_H

#include <iostream>
#include<fstream>
#include <string>
#include <vector>
#include<queue>
#include<set>
#include<map>
using namespace std;


enum tokenKind {
	KEY,	//保留字
	SYM,	//特殊符号
	ID,		//标识符
	NUM,	//数值常数
	STR,	//字符串常数
	TK_TRUE,// 细化KEY
	TK_FALSE,
	TK_OR,
	TK_AND,
	TK_NOT,
	TK_INT,
	TK_BOOL,
	TK_STRING,
	TK_WHILE,
	TK_DO,
	TK_IF,
	TK_THEN,
	TK_ELSE,
	TK_END,
	TK_REPEAT,
	TK_UNTIL,
	TK_READ,
	TK_WRITE,
	TK_GTR,	// 细化操作符
	TK_LEQ,
	TK_GEQ,
	TK_COMMA,
	TK_SEMICOLON,
	TK_ASSIGN,
	TK_ADD,
	TK_SUB,
	TK_MUL,
	TK_DIV,
	TK_LP,
	TK_RP,
	TK_LSS,
	TK_EQU,
	END_OF_FILE,
};


class Token {
	private:
		// 各个token类别对应的名字，因为enum中无法得到起名字
		static vector<string> kind_names_;
		// 该token的类别
		tokenKind kind;
		// 该token的值，如NUM类的token可以是'123'、'23'
		string value;
		// 该token所处的位置
		int line;
		unsigned int column;

	public:
		// 定义构造函数
		Token();
		Token(tokenKind kind, string value, int line = 0, unsigned int column = 0);

		// accesser
		tokenKind GetKind();
		string GetValue();
		int GetLine();
		unsigned long GetColumn();
		string GetKindName();
		static string GetKindName(tokenKind kind);

		static bool is_KEY(string& str);
};


class Tokens {
private:
	vector<Token> tokens_;

public:
	void push(Token token);

	void push(tokenKind kind, string value, int line = 0, unsigned long column = 0);

	void clear();

	void append(Tokens tokens);

	vector<Token>::iterator begin();

	vector<Token>::iterator end();

	Token back();

	unsigned long size() const;

	void RefineKind();
};


class ErrorMsg {
private:
	int line;
	int column;
	string msg;

public:
	ErrorMsg(int line_, int column_, string msg_) : line(line_), column(column_), msg(msg_) {}

	void print();

	string to_string();
};

using ErrorMsgs = vector<ErrorMsg>;


#endif

