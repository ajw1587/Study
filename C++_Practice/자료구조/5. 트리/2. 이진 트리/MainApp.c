#include "ArrayBaseBTree.h"
#include "NodeBaseTree.h"
void ArrayBaseTreeTest();
void NodeBaseTreeTest();
int main() {
	NodeBaseTreeTest();

	return 0;
}
void NodeBaseTreeTest() {
	bst* bst1 = initBSTree();
	bst_addNode(bst1, 10);
	bst_addNode(bst1, 7);
	bst_addNode(bst1, 8);
	bst_addNode(bst1, 5);
	bst_addNode(bst1, 85);
	bst_addNode(bst1, 53);
	bst_addNode(bst1, 10);

	bst_Orderring(bst1, bst_PreOrder);
	printf("\n");
	bst_Orderring(bst1, bst_InOrder);
	printf("\n");
	bst_Orderring(bst1, bst_PostOrder);
	printf("\n");
}


void ArrayBaseTreeTest() {
	btree* bt = initBTree(TREE_SIZE);
	addData(bt, 10);
	addData(bt, 20);
	addData(bt, 30);
	addData(bt, 22);
	addData(bt, 63);
	addData(bt, 78);
	addData(bt, 45);
	addData(bt, 34);
	addData(bt, 24);

	PreOrder(bt, 1);
}