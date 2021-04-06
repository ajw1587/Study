#include "SingleLinkedList.h"

typedef slist list;


int main() {
	list* l1 = createList(10);
	int menu = 0;
	int data = 0;
	int position = 0;
	while (menu != 5) {
		printf("list elements > ");
		viewList(l1);
		printf("\n");
		printf("1. append  2.insert  3.delete  ");
		printf("4. search  5. exit\nselect > ");
		scanf("%d", &menu);

		switch (menu) {
		case 1: 
			printf("input data to append > ");
			scanf("%d", &data);
			appendList(l1, data); 
			break;
		case 2:
			printf("input  insert Position > ");
			scanf("%d", &position);
			printf("input data to insert > ");
			scanf("%d", &data);
			insertNode(l1, position, data);
			break;
		case 3:
			printf("input  delete Position > ");
			scanf("%d", &position);
			deleteNode(l1, position);
			break;
		}
	}


}