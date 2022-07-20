#include "intermediate_code.h"

// 获取新的标号
string IntermediateCode::get_new_label() {
	return "L" + to_string(label_num++);
}

// 获取新的变量名
string IntermediateCode::get_var_name() {
	return "t" + to_string(var_num++);
}

// 对语法分析树进行分析
void IntermediateCode::stmt_node(TreeNode* root) {
	if (!root)
		return;
	string after_stmt_label = get_new_label();

	switch (root->type_) {
	case TreeNode::Type::IF_STMT:
		// 对if进行分析
		if_node(root, after_stmt_label);
		break;
	case TreeNode::Type::ASSIGN_STMT:
		assign_node(root);
		break;
	case TreeNode::Type::WHILE_STMT:
		while_node(root, after_stmt_label);
		break;
	case TreeNode::Type::REPEAT_STMT:
		repeat_node(root, after_stmt_label);
		break;
	case TreeNode::Type::READ_STMT:
		read_node(root);
		break;
	case TreeNode::Type::WRITE_STMT:
		write_node(root);
		break;
	default:
		break;
	}
	res_.push_back("Label " + after_stmt_label);
}

// 处理if节点
void IntermediateCode::if_node(TreeNode* root, string end_label) {
	string before_else_label = end_label;
	string before_then_label = get_new_label();
	// 判断是否有else的内容
	if (root->child_[2]) {	
		before_else_label = get_new_label();
		cond_exp(root->child_[0], before_then_label, before_else_label);
		res_.push_back("Label " + before_then_label);
		stmt_seq_node(root->child_[1]);
		res_.push_back("goto " + end_label);
		res_.push_back("Label " + before_else_label);
		stmt_seq_node(root->child_[2]);
	}
	else {
		cond_exp(root->child_[0], before_then_label, end_label);
		res_.push_back("Label " + before_then_label);
		stmt_seq_node(root->child_[1]);
		res_.push_back("Label " + end_label);
	}
}

// 处理或运算逻辑
string IntermediateCode::log_or(TreeNode* root) {

	switch (root->type_) {
	case TreeNode::Type::LOG_OR_EXP: {
		string log_and_res = log_and(root->child_[0]);
		string another_log_or_res = log_or(root->child_[1]);
		string log_or_res = get_var_name();
		res_.push_back(log_or_res + ":=" + log_and_res + "or" + another_log_or_res);
		return log_or_res;
	}
	default:
		return log_and(root);
	}
}

// 处理与运算逻辑
string IntermediateCode::log_and(TreeNode* root) {
	switch (root->type_) {
	case TreeNode::Type::LOG_AND_EXP: {
		string comp_exp_res = comparison_exe(root->child_[0]);
		string second_log_and_res = log_and(root->child_[1]);
		string log_and_res = get_var_name();
		res_.push_back(log_and_res + ":=" + comp_exp_res + "and" + second_log_and_res);
		return log_and_res;
	}
	default:
		return comparison_exe(root);
	}
}

// 比较运算符生成代码
string IntermediateCode::comparison_exe(TreeNode* root) {
	switch (root->type_) {
	case TreeNode::Type::LEQ_EXP:
	case TreeNode::Type::LSS_EXP:
	case TreeNode::Type::GTR_EXP:
	case TreeNode::Type::GEQ_EXP:
	case TreeNode::Type::EQU_EXP: {
		string comp_left = add_exp(root->child_[0]);
		string comp_right = comparison_exe(root->child_[1]);
		string op;
		switch (root->type_) {
		case TreeNode::Type::LEQ_EXP:
			op = "<=";
			break;
		case TreeNode::Type::LSS_EXP:
			op = "<";
			break;
		case TreeNode::Type::GTR_EXP:
			op = ">";
			break;
		case TreeNode::Type::GEQ_EXP:
			op = ">=";
			break;
		case TreeNode::Type::EQU_EXP:
			op = "=";
			break;
		default:
			break;
		}
		return comp_left + op + comp_right;
	}
	default:
		return add_exp(root);
	}
}

// 分析加法和减法的运算
string IntermediateCode::add_exp(TreeNode* root) {
	switch (root->type_) {
	case TreeNode::Type::ADD_EXP:
	case TreeNode::Type::SUB_EXP: {
		// 递归分析左右子树
		string add_left = mul_exe(root->child_[0]);
		string add_right = add_exp(root->child_[1]);
		string add_res = get_var_name();
		string op;
		if (root->type_ == TreeNode::Type::ADD_EXP)
			op = "+";
		if (root->type_ == TreeNode::Type::SUB_EXP)
			op = "-";
		res_.push_back(add_res + ":=" + add_left + op + add_right);
		return add_res;
	}
	default:
		return mul_exe(root);
	}
}

// 分析乘法和除法运算
string IntermediateCode::mul_exe(TreeNode* root) {
	switch (root->type_) {
	case TreeNode::Type::MUL_EXP:
	case TreeNode::Type::DIV_EXP: {
		// 递归分析左右子树
		string mul_left = mul_exe(root->child_[0]);
		string mul_right = add_exp(root->child_[1]);
		string mul_res = get_var_name();
		string op;
		if (root->type_ == TreeNode::Type::MUL_EXP)
			op = "*";
		if (root->type_ == TreeNode::Type::DIV_EXP)
			op = "/";
		res_.push_back(mul_res + " := " + mul_left + op + mul_right);
		return mul_res;
	}
	default:
		return factor(root);
	}
}

// 处理因子 直接返回对应因子的值
string IntermediateCode::factor(TreeNode* root) {
	switch (root->tk_.GetKind()) {
	case tokenKind::ID:
		return root->tk_.GetValue();
	case tokenKind::NUM:
		return root->tk_.GetValue();
	case tokenKind::STR:
		return "'" + root->tk_.GetValue() + "'";
	case tokenKind::TK_FALSE:
		return "false";
	case tokenKind::TK_TRUE:
		return "true";
	default:
		return log_or(root);
	}
}

// 条件语句分析
void IntermediateCode::cond_exp(TreeNode* root, string true_label, string false_label) {

	switch (root->type_) {
		// 如果当前节点是 or 就递归的分析左右节点
		case TreeNode::Type::LOG_OR_EXP: {
			string before_second_log_or_label = get_new_label();
			cond_exp(root->child_[0], true_label, before_second_log_or_label);
			res_.push_back("Label " + before_second_log_or_label);
			cond_exp(root->child_[1], true_label, false_label);
			break;
		}
		// 如果当前节点是 and 同样递归的分析左右节点
		case TreeNode::Type::LOG_AND_EXP: {
			string before_second_log_and_label = get_new_label();
			cond_exp(root->child_[0], before_second_log_and_label, false_label);
			res_.push_back("Label " + before_second_log_and_label);
			cond_exp(root->child_[1], true_label, false_label);
			break;
		}
		// 如果当前节点是不是and或者or，说明可能是比较运算符或者运算符 从比较运算符开始递归分析
		default: {
			string comp_res = comparison_exe(root);
			res_.push_back("if " + comp_res + " goto " + true_label);
			res_.push_back("goto " + false_label);
			break;
		}
	}
}

// 赋值运算符，左边只能是一个变量名，右边可能是一个表达式 递归分析
void IntermediateCode::assign_node(TreeNode* root) {
	string log_or_res = log_or(root->child_[1]);
	string id_res = factor(root->child_[0]);
	res_.push_back(id_res + ":=" + log_or_res);
}

// 递归生成while代码
void IntermediateCode::while_node(TreeNode* root, string end_label) {
	string before_cond_label = get_new_label();
	string before_then_label = get_new_label();
	res_.push_back("Label " + before_cond_label);
	cond_exp(root->child_[0], before_then_label, end_label);
	res_.push_back("Label " + before_then_label);
	stmt_seq_node(root->child_[1]);
	res_.push_back("goto " + before_cond_label);
}

// 递归生成read代码
void IntermediateCode::read_node(TreeNode* root) {
	res_.push_back("read " + root->tk_.GetValue());
}

// 递归生成write代码
void IntermediateCode::write_node(TreeNode* root) {
	res_.push_back("write " + root->child_[0]->tk_.GetValue());
}

// 递归生成repeat代码
void IntermediateCode::repeat_node(TreeNode* root, string end_label) {
	string before_stmt_label = get_new_label();
	res_.push_back("Label " + before_stmt_label);
	stmt_seq_node(root->child_[0]);
	cond_exp(root->child_[1], end_label, before_stmt_label);
}

// 入口函数
void IntermediateCode::stmt_seq_node(TreeNode* root) {
	switch (root->type_) {
	case TreeNode::Type::STMT_SEQUENCE:
		stmt_seq_node(root->child_[0]);
		stmt_node(root->child_[1]);
		break;
	default:
		stmt_node(root);
	}
}

