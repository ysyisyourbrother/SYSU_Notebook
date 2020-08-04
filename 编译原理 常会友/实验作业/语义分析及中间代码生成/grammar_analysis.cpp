#include "grammar_analysis.h"


// 定义构造函数
GrammarAnalysis::GrammarAnalysis(Tokens tokens) {
	tokens_ = tokens;
	token_it_ = tokens_.begin();	// 记录当前分析的token位置
}

// 语法分析程序入口，同时收集异常信息
TreeNode* GrammarAnalysis::program(ErrorMsgs& error_msgs) {
	if (tokens_.size() == 0)
		return NULL;
	try {
		declaration();	// 先将声明部分跳过
		return stmt_sequence();	// 开始构建树
	}
	catch (ErrorMsg & msg) {
		std::cout << msg.to_string() << std::endl;
		error_msgs.emplace_back(msg);
		return NULL;
	}
	return NULL;
}


// tiny 语法中声明都要放在前面，这些不作为语法分析的内容
void GrammarAnalysis::declaration() {
	// 跳过类型声明符号int、bool、string
	while (match({ tokenKind::TK_INT, tokenKind::TK_BOOL, tokenKind::TK_STRING })) {
		// 记录当前token的类别
		ValType val_type;
		if (token_it_->GetKind() == tokenKind::TK_INT)
			val_type = ValType::VT_INT;
		if (token_it_->GetKind() == tokenKind::TK_BOOL)
			val_type = ValType::VT_BOOL;
		if (token_it_->GetKind() == tokenKind::TK_STRING)
			val_type = ValType::VT_STRING;

		// 循环匹配多个变量，不同变量用逗号分隔
		do {
			++token_it_;
			match({ tokenKind::ID }, "类型声明符号后面需要跟变量名");

			// 在符号表中插入该变量
			Sym* sym = sym_table_.insert(token_it_->GetValue());
			if (sym == NULL) throw ErrorMsg(token_it_->GetLine(), token_it_->GetColumn(), "标识符已经存在");
			// 记录变量类型variable
			sym->obj_type = ObjType::OT_VAR;

			// 记录值类型 int bool string
			sym->val_type = val_type;

			// 记录当前token
			sym->tk = *token_it_;	

			token_it_++;
		} while (match({ tokenKind::TK_COMMA })); //匹配逗号
		match({ tokenKind::TK_SEMICOLON }, "变量声明最后应该加';'"); //匹配分号
		++token_it_;
	}
}


TreeNode* GrammarAnalysis::stmt_sequence() {
	TreeNode* t1 = NULL, * t2 = NULL;
	// tiny语言都是有特定的几种符号开始，先匹配上这些，然后继续构建语法树
	KindSet first_statement = { tokenKind::TK_IF, tokenKind::TK_WHILE, tokenKind::TK_REPEAT,tokenKind::ID, tokenKind::TK_READ, tokenKind::TK_WRITE };

	while (match(first_statement)) {
		switch ((*token_it_).GetKind()) {
		case tokenKind::TK_IF:
			t2 = if_stmt();	// 如果当前token为if，调用if的分析函数
			break;
		case tokenKind::TK_WHILE:
			t2 = while_stmt();
			break;
		case tokenKind::TK_REPEAT:
			t2 = repeat_stmt();
			break;
		case tokenKind::ID:
			t2 = assign_stmt();
			break;
		case tokenKind::TK_READ:
			t2 = read_stmt();
			break;
		case tokenKind::TK_WRITE:
			t2 = write_stmt();
			break;
		default:
			break;
		}
		if (t1 == NULL)
			t1 = t2;
		else
			t1 = new TreeNode(TreeNode::Type::STMT_SEQUENCE, t1, t2);
	}
	return t1;
}

// 处理if情况
TreeNode* GrammarAnalysis::if_stmt() {
	match({ tokenKind::TK_IF }, "期望是if关键词");
	++token_it_;	// 进入if的下一个符号，正常情况是一个条件

	// if语句存在三种分支可能，分别为条件、正分支和错分支，需要定义三个树节点
	TreeNode* condition_exp = NULL, * then_stmt = NULL, * else_stmt = NULL;

	// 先计算条件节点的内容
	condition_exp = log_or_exp();

	// 计算正分支的内容
	match({ tokenKind::TK_THEN }, "期望是then关键词");
	++token_it_;
	then_stmt = stmt_sequence();

	// 错分支不是必须的，如果不存在第三个分支为空
	if (match({ tokenKind::TK_ELSE })) {
		++token_it_;
		else_stmt = stmt_sequence();
	}

	// 语义检查 如果不是bool类型的抛出异常
	std::cout << condition_exp->val_type_;
	if (condition_exp->val_type_ != ValType::VT_BOOL)
		throw ErrorMsg(token_it_->GetLine(), token_it_->GetColumn(), "if的条件表达式的值必须是bool类型");

	match({ tokenKind::TK_END }, "if块缺少end结束符");
	++token_it_;

	return new TreeNode(TreeNode::Type::IF_STMT, condition_exp, then_stmt, else_stmt);
}

// 处理while情况
TreeNode* GrammarAnalysis::while_stmt() {
	match({ tokenKind::TK_WHILE }, "缺少while关键词");
	++token_it_;
	// while的循环条件
	TreeNode* log_or_exp_node = log_or_exp();

	match({ tokenKind::TK_DO }, "while块缺少do关键词");
	++token_it_;
	// while循环体的内容
	TreeNode* stmt_seq_node = stmt_sequence();

	match({ tokenKind::TK_END }, "while块缺少end关键词");
	++token_it_;

	// 当前节点是while，左孩子是循环条件，右孩子是循环体。t3留空。
	return new TreeNode(TreeNode::Type::WHILE_STMT, log_or_exp_node, stmt_seq_node);
}



TreeNode* GrammarAnalysis::repeat_stmt() {
	match({ tokenKind::TK_REPEAT }, "Repeat块缺少repeat关键词");
	++token_it_;

	TreeNode* stmt_seq_node = stmt_sequence();

	match({ tokenKind::TK_UNTIL }, "Repeat块缺少until关键词");
	++token_it_;

	// until语句的结束条件
	TreeNode* log_or_exp_node = log_or_exp();

	// 当前节点是repeat，左孩子是until里的命令，右孩子是结束条件。t3留空。
	return new TreeNode(TreeNode::Type::REPEAT_STMT, stmt_seq_node, log_or_exp_node);
}



// 处理赋值情况
TreeNode* GrammarAnalysis::assign_stmt() {
	TreeNode* node = factor();

	if (node->tk_.GetKind() != tokenKind::ID)
		throw ErrorMsg(token_it_->GetLine(), token_it_->GetColumn(), "赋值语句左边应该是变量");

	if (match({ tokenKind::TK_ASSIGN }, "赋值语句应该有:=")) {
		token_it_++;
		// 赋值语句后面的表达式
		TreeNode* log_or_exp_node = log_or_exp();

		// 加入判断 赋值等号两边需要为相同数据类型
		if (node->val_type_ != log_or_exp_node->val_type_)
			throw ErrorMsg(token_it_->GetLine(), token_it_->GetColumn(), "赋值语句左右两边应该是同种类型");

		match({ tokenKind::TK_SEMICOLON }, "赋值语句最后应有分号");
		++token_it_;

		return new TreeNode(TreeNode::Type::ASSIGN_STMT, node, log_or_exp_node);
	}
	return NULL;
}

// 分析因子
TreeNode* GrammarAnalysis::factor() {
	KindSet factor_set = { tokenKind::NUM, tokenKind::STR, tokenKind::ID, tokenKind::TK_TRUE, tokenKind::TK_FALSE };
	TreeNode* mul_exp_node = NULL;

	// (TRUE)这样也是一个因子，需要递归考虑
	if (match({ tokenKind::TK_LP })) {
		token_it_++; // 分析完因子节点后，指针后移
		TreeNode* log_or_exp_node = log_or_exp();
		match({ tokenKind::TK_RP }, "'('缺少')'的匹配");
		++token_it_;
		return log_or_exp_node;
	}
	// 因子可能的形式为数字，字符串，变量以及true和false
	else if (match(factor_set, "匹配数字,字符串,变量,true,false失败")) {
		// 定义因子节点
		TreeNode* factor_node = new TreeNode(TreeNode::Type::FACTOR, *token_it_);

		// 从符号表中获取类型
		if (match({ tokenKind::ID })) {
			// 判断ID是否在使用之前进行了声明，如果没有声明抛出异常
			if (!sym_table_.find((*token_it_).GetValue()))
				throw ErrorMsg((*token_it_).GetLine(), (*token_it_).GetColumn(), "使用了未声明的变量");
			else {
				Sym* sym = sym_table_.find((*token_it_).GetValue());
				factor_node->val_type_ = sym->val_type;
			}
		}

		// 给因子节点添加实际的数据类型
		if (match({ tokenKind::NUM }))
			factor_node->val_type_ = ValType::VT_INT;

		else if (match({ tokenKind::STR }))
			factor_node->val_type_ = ValType::VT_STRING;

		else if (match({ tokenKind::TK_TRUE, tokenKind::TK_FALSE }))
			factor_node->val_type_ = ValType::VT_BOOL;
		++token_it_;
		return factor_node;
	}
	return NULL;
}


// 逻辑或运算
TreeNode* GrammarAnalysis::log_or_exp() {
	TreeNode* log_and_exp_node = NULL, * log_or_exp_node = NULL;
	log_and_exp_node = log_and_exp();// 类似逆波兰表达式，求逻辑或时，先求优先级大一级的，即逻辑与。

	if (match({ tokenKind::TK_OR })) {
		token_it_++;
		// 递归解决多个连续的或运算的情况
		log_or_exp_node = log_or_exp();
	}
	// 若有两个节点，则产生新的或节点，若只有一个，则直接返回这一个节点
	if (log_or_exp_node) {
		TreeNode* tmp = new TreeNode(TreeNode::Type::LOG_OR_EXP, log_and_exp_node, log_or_exp_node);
		tmp->val_type_ = ValType::VT_BOOL;	// 定义返回值类型为布尔值
		return tmp;
	}
	else
		return log_and_exp_node;
}

// 逻辑与运算
TreeNode* GrammarAnalysis::log_and_exp() {
	// and优先级小于comparision
	TreeNode* log_and_exp_node = comparision_exp();
	TreeNode* another_log_and_exp_node = NULL;

	// 后半部分的E可能是由多个and组成，递归处理
	if (match({ tokenKind::TK_AND })) {
		token_it_++;
		another_log_and_exp_node = log_and_exp();
	}

	if (another_log_and_exp_node) {
		TreeNode* tmp = new TreeNode(TreeNode::Type::LOG_AND_EXP, log_and_exp_node, another_log_and_exp_node);
		tmp->val_type_ = ValType::VT_BOOL;
		return tmp;
	}
	else
		return log_and_exp_node;
}

// 处理比较运算符情况
TreeNode* GrammarAnalysis::comparision_exp() {
	// comparision的优先级小于add
	TreeNode* add_exp_node = add_exp();

	KindSet comparison_op_set = { tokenKind::TK_LEQ, tokenKind::TK_GEQ, tokenKind::TK_LSS, tokenKind::TK_GTR, tokenKind::TK_EQU };

	if (match(comparison_op_set)) {
		Token prev_token = *token_it_;
		token_it_++;
		// 定义比较节点
		TreeNode* comparison_node = new TreeNode();
		if (prev_token.GetKind() == tokenKind::TK_LEQ)
			comparison_node = new TreeNode(TreeNode::Type::LEQ_EXP, add_exp_node, comparision_exp());
		if (prev_token.GetKind() == tokenKind::TK_GEQ)
			comparison_node = new TreeNode(TreeNode::Type::GEQ_EXP, add_exp_node, comparision_exp());
		if (prev_token.GetKind() == tokenKind::TK_LSS)
			comparison_node = new TreeNode(TreeNode::Type::LSS_EXP, add_exp_node, comparision_exp());
		if (prev_token.GetKind() == tokenKind::TK_EQU)
			comparison_node = new TreeNode(TreeNode::Type::EQU_EXP, add_exp_node, comparision_exp());
		if (prev_token.GetKind() == tokenKind::TK_GTR)
			comparison_node = new TreeNode(TreeNode::Type::GTR_EXP, add_exp_node, comparision_exp());
		comparison_node->val_type_ = ValType::VT_BOOL;

		return comparison_node;
	}
	return add_exp_node;
}

// 处理加法情况
TreeNode* GrammarAnalysis::add_exp() {
	// add的优先级小于mul
	TreeNode* add_exp_node = mul_exp();

	KindSet add_op_set = { tokenKind::TK_ADD, tokenKind::TK_SUB };
	if (match(add_op_set)) {
		Token prev_token = *token_it_;
		token_it_++;

		TreeNode* another_add_exp_node = add_exp();

		if (prev_token.GetKind() == tokenKind::TK_ADD) {
			TreeNode* tmp = new TreeNode(TreeNode::Type::ADD_EXP, add_exp_node, another_add_exp_node);
			tmp->val_type_ = add_exp_node->val_type_;
			return tmp;
		}
		else if (prev_token.GetKind() == tokenKind::TK_SUB) {
			TreeNode* tmp = new TreeNode(TreeNode::Type::SUB_EXP, add_exp_node, another_add_exp_node);
			tmp->val_type_ = add_exp_node->val_type_;
			return tmp;
		}
	}
	return add_exp_node;
}


TreeNode* GrammarAnalysis::mul_exp() {
	// mul的优先级小于factor
	TreeNode* node1 = factor();

	if (token_it_->GetKind() == tokenKind::TK_MUL) {
		token_it_++;
		TreeNode* node2 = mul_exp();

		// 当前节点是乘，左右孩子分别是两个操作数
		TreeNode* node3 = new TreeNode(TreeNode::Type::MUL_EXP, node1, node2);
		node3->val_type_ = ValType::VT_INT;
		return node3;
	}
	else if (token_it_->GetKind() == tokenKind::TK_DIV) {
		token_it_++;
		TreeNode* node2 = mul_exp();

		// 当前节点是除，左右孩子分别是两个操作数
		TreeNode* node3 = new TreeNode(TreeNode::Type::DIV_EXP, node1, node2);
		node3->val_type_ = ValType::VT_INT;
		return node3;
	}
	// 当前节点就是单一的数/表达式，不需要拓展子节点
	else return node1;
}

// 处理read的情况
TreeNode* GrammarAnalysis::read_stmt() {
	match({ tokenKind::TK_READ }, "read块缺少read关键词");
	++token_it_;

	if (match({ tokenKind::ID }, "read 后面跟变量")) {
		token_it_++;

		TreeNode* read_node = new TreeNode(TreeNode::Type::READ_STMT, *token_it_);

		match({ tokenKind::TK_SEMICOLON }, "read语句最后以';'结尾");
		++token_it_;

		return read_node;
	}
	return NULL;
}

// 处理write情况
TreeNode* GrammarAnalysis::write_stmt() {
	match({ tokenKind::TK_WRITE }, "write块缺少write关键词");
	++token_it_;

	TreeNode* log_or_exp_node = log_or_exp();

	match({ tokenKind::TK_SEMICOLON }, "write语句最后以';'结尾");
	++token_it_;

	return new TreeNode(TreeNode::Type::WRITE_STMT, log_or_exp_node);
}

// 判断是否匹配结果
bool GrammarAnalysis::match(KindSet kind_set, string eof_error_msg)  throw(ErrorMsg) {
	// 判断是否token分析完成，并抛出异常
	if (eof_error_msg != "") {
		if (token_it_ == tokens_.end())
			throw ErrorMsg(tokens_.back().GetLine(), tokens_.back().GetColumn(), eof_error_msg);
		else if (kind_set.find((*token_it_).GetKind()) == kind_set.end())
			throw ErrorMsg((*token_it_).GetLine(), (*token_it_).GetColumn(), eof_error_msg);
		return true;
	}
	// 若错误信息为空，说明已经分析完成
	else if (token_it_ == tokens_.end())
		return false;

	return kind_set.find((*token_it_).GetKind()) != kind_set.end();
}

// 按照层序遍历打印语法树
void GrammarAnalysis::print_tree() {
	if (!root) return;
	queue<TreeNode*> que;
	que.push(root);
	while (!que.empty()) {
		int level_len = que.size();
		for (int i = 0; i < level_len; i++) {
			TreeNode* curr = que.front();
			que.pop();
			if (curr) {
				cout << curr->get_type_name() << " ";
				for (auto child : curr->child_) que.push(child);
			}
			else cout << "null" << " ";
		}
		cout << endl;
	}
	return;
}

// 打印语义分析的符号表
void GrammarAnalysis::print_sym_table() {
	sym_table_.print();
}
