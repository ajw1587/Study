#include "AVLtree.h"
Avltree* initAvl(void) {
	Avltree* temp = (Avltree*)malloc(sizeof(Avltree));
	temp->root = NULL;
	return temp;
}

node* LL_Rotate(node* pnode) {
	node* parent = pnode;
	node* child = parent->left;

	parent->left = child->right;
	child->right = parent;

	return child;
}

node* RR_Rotate(node* pnode) {
	node* parent = pnode;
	node* child = parent->right;

	parent->right = child->left;
	child->left = parent;
	return child;
}


node* LR_Rotate(node* pnode) {
	node* parent = pnode;
	pnode->left = RR_Rotate(pnode->left);
	return LL_Rotate(pnode);
}
node* RL_Rotate(node* pnode) {
	node* parent = pnode;
	pnode->right = LL_Rotate(pnode->right);
	return RR_Rotate(pnode);

}

int getHeight(node* pnode) {
	if (pnode == NULL) return 0;
	int hLeft = getHeight(pnode->left);
	int hRight = getHeight(pnode->right);

	return hLeft > hRight ? hLeft + 1 : hRight + 1;
}


int heightDiff(node* pnode) {
	if (pnode == NULL) return 0;
	int hLeft = getHeight(pnode->left);
	int hRight = getHeight(pnode->right);

	return hLeft - hRight;
}

void ReBalance(Avltree* _block) {
	int hDiff = heightDiff(_block->root);
	if (hDiff > 1) {
		if (heightDiff(_block->root->left) > 0) {
			_block->root = LL_Rotate(_block->root);
		}
		else
			_block->root = LRRotate(_block->root);
	}

	if (hDiff < -1) {
		if (heightDiff(_block->root->right) < 0) {
			_block->root = RR_Rotate(_block->root);
		}
		else _block->root = RL_Rotate(_block->root);
	}
}



void insertData(Avltree* block, int _key) {
	node* cursor = NULL;
	if (block->root == NULL) {
		block->root = initNode();
		setData(block->root, _key);
		return;
	}
	cursor = block->root;
	while (1) {
		if (cursor->data == _key) {
			printf("< %d > is Already exist\n");
			return;
		}
		else if (cursor->data > _key) {
			if (cursor->left == NULL) {
				cursor->left = initNode();
				setData(cursor->left, _key);
				break;
			}
			else {
				cursor = cursor->left;
				continue;
			}
		}
		else { // cursor ->data   < _key
			if (cursor->right == NULL) {
				cursor->right = initNode();
				setData(cursor->right, _key);
				break;
			}
			else {
				cursor = cursor->right;
				continue;
			}
		}
	}

	ReBalance(block);

}





typedef void orderfunc(node*);

void orderring(Avltree*, orderfunc order);

void PreOrder(node*);
void PostOrder(node*);
void InOrder(node*);