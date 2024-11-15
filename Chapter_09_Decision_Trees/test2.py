from sklearn.tree import DecisionTreeClassifier

# Example data
X = [[0, 0], [1, 1]]
y = [0, 1]

dt = DecisionTreeClassifier()
dt.fit(X, y)

# Display the tree
display_tree(dt)
