#pragma once
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
	// 细化后的token类型
	TK_TRUE,
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
	TK_GTR,
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

	// 细化token的类别。
	void RefineKind();
};

// 定义tokens序列类型
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

// 定义错误信息类型
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

// 定义值的类型
enum ValType {
	VT_INT,         // 整型数类型
	VT_BOOL,        // 布尔类型
	VT_STRING,      // 字符串类型
	VT_UNDEFINED	// 为了初始化TreeNode新增的哑属性
};

// 定义对象类型
enum ObjType {
	OT_FUN,         // 函数
	OT_VAR,         // 变量
	OT_CONST,       // 常量
};


class TreeNode {
private:
	// 和Token类一样，因为用enum定义节点类别不能获得字符串，所以加上这个
	static vector<string> type_names;

public:
	enum Type {
		PROGRAM,        // 程序（开始符号）节点
		STMT_SEQUENCE,  // 语句列表节点
		IF_STMT,        // 条件语句节点
		REPEAT_STMT,    // repeat语句节点
		ASSIGN_STMT,    // 赋值语句节点
		READ_STMT,      // read语句节点
		WRITE_STMT,     // write语句节点
		WHILE_STMT,     // while语句节点
		GTR_EXP,        // 大于表达式节点
		GEQ_EXP,        // 大于等于表达式节点
		LSS_EXP,        // 小于表达式节点
		LEQ_EXP,        // 小于等于表达式节点
		EQU_EXP,        // 等于表达式节点
		LOG_OR_EXP,     // 逻辑或表达式节点
		LOG_AND_EXP,    // 逻辑与表达式节点
		LOG_NOT_EXP,    // 逻辑非表达式节点
		ADD_EXP,        // 加法表达式节点
		SUB_EXP,        // 减法表达式节点
		MUL_EXP,        // 乘法表达式节点
		DIV_EXP,        // 除法表达式节点
		FACTOR          // 原子节点
	};
	Type type_;	// 节点类型
	ValType val_type_ = VT_UNDEFINED;	//节点值类型 ValType
	vector<TreeNode*>child_ = { NULL,NULL,NULL };
	Token tk_;

	// 默认构造函数
	TreeNode(){}

	// 语法分析树的内部节点最多有三个子节点，有的少于三个就把t3留空
	TreeNode(Type type, TreeNode* t1 = NULL, TreeNode* t2 = NULL, TreeNode* t3 = NULL);

	// 语法分析树的叶节点，直接拓展为token，所以不需要考虑子节点了
	TreeNode(Type type, Token token);

	string get_type_name();
	static string type_name(Type type);
};




#endif

