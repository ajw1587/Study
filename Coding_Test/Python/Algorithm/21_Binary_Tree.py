class Node:
    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None

class BinaryTree():
    def __init__(self):
        self.root = None
    
    # 전위 순회
    def preorder(self, n):
        if n != None:
            print(n.item, ' ', end = '')
            if n.left:
                self.preorder(n.left)
            if n.right:
                self.preorder(n.right)
    
    # 중위 순회
    def inorder(self, n):
        if n != None:
            if n.left:
                self.preorder(n.left)
            print(n.item, ' ', end = '')
            if n.right:
                self.preorder(n.right)
    
    # 후위 순회
    def postorder(self, n):
        if n != None:
            if n.left:
                self.preorder(n.left)
            if n.right:
                self.preorder(n.right)
            print(n.item, ' ', end = '')
    
    # 레벨 순회
    def levelorder(self, root):
        q = []
        q.append(root)
        while q:
            t = q.pop(0)
            print(t.item, ' ', end = '')
            if t.left != None:
                q.append(t.left)
            if t.right != None:
                q.append(t.right)

    # 높이
    def height(self, root):
        if root == None:
            return 0
        return max(self.height(root.left), self.height(root.right)) + 1

tree = BinaryTree()
n1 = Node(10)
n2 = Node(20)
n3 = Node(30)
n4 = Node(40)
n5 = Node(50)
n6 = Node(60)
n7 = Node(70)
n8 = Node(80)

# 트리 만들기
tree.root = n1
n1.left = n2
n1.right = n3
n2.left = n4
n2.right = n5
n3.left = n6
n3.right = n7
n4.left = n8

# 출력>> 트리 높이: 4
print('트리 높이: ', tree.height(tree.root))

# 출력>> 전위 순회: 10  20  40  80  50  30  60  70
print('전위 순회: ', end = '')
tree.preorder(tree.root)

# 출력>> 중위 순회: 20  40  80  50  10  30  60  70
print('\n중위 순회: ', end = '')
tree.inorder(tree.root)

# 출력>> 후위 순회: 20  40  80  50  30  60  70  10
print('\n후위 순회: ', end = '')
tree.postorder(tree.root)

# 출력>> 레벨 순회: 10  20  30  40  50  60  70  80
print('\n레벨 순회: ', end = '')
tree.levelorder(tree.root)
