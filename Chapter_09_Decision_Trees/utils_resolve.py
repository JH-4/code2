import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from io import StringIO
import pydotplus
from IPython.display import Image

# Prepare the dataset
dataset = pd.DataFrame({
    'x_0': [7, 3, 2, 1, 2, 4, 1, 8, 6, 7, 8, 9],
    'x_1': [1, 2, 3, 5, 6, 7, 9, 10, 5, 8, 4, 6],
    'y': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})

features = dataset[['x_0', 'x_1']]
labels = dataset['y']

# Train the decision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(features, labels)

# Score the model
print("Model Accuracy:", decision_tree.score(features, labels))

# Display the decision tree
def display_tree(dt, feature_names, output_file="decision_tree.png"):
    dot_data = StringIO()
    export_graphviz(dt, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(output_file)
    print(f"Decision tree saved as {output_file}")

# Save and display the decision tree
display_tree(decision_tree, features.columns)
