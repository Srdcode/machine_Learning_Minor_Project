import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



def main():
    st.title("My app")


    # Sidebar navigation
    page = st.sidebar.selectbox("Select Page", ["Home", "Documentation"])

    if page == "Home":
        prediction_page()
    elif page == "Documentation":
        documentation_page()

def prediction_page():
    st.subheader("Home / Prediction Page")
    # Add your prediction-related content here
    # For example, you can add input widgets and prediction logic
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()

    # # Load the saved model using pickle
    # model_filename = 'decision_tree_model.pkl'
    # with open(model_filename, 'rb') as model_file:
    #     loaded_model = pickle.load(model_file)

    # Load the saved models using pickle
    model_filenames = {
        'Decision Tree': 'decision_tree_model.pkl',
        'KNN': 'knn.pkl',
        'Logistic Regression': 'Logistic_Regression.pkl',
        'Random Forest': 'random_forest_model.pkl'
    }

    loaded_models = {model_name: pickle.load(open(filename, 'rb')) for model_name, filename in model_filenames.items()}

    # Load the unique values for categorical columns
    categorical_values = {
        'Workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'unknown'],
        'Education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
        'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'unknown'],
        'sex': ['Male', 'Female']
    }

    # Streamlit app
    st.title("Income Prediction App")

    # User input for features
    feature_inputs = {}
    feature_inputs['Age'] = st.number_input("Enter Age:", value=0)

    for col in ['Workclass', 'Education', 'occupation', 'sex']:
        selected_option = st.selectbox(f"Select {col}:", categorical_values[col])
        feature_inputs[col] = selected_option
        encoder.fit(categorical_values[col])  # Fit the encoder on the unique values
        encoded_value = encoder.transform([selected_option])[0]
        feature_inputs[col] = encoded_value

    feature_inputs['hours_per_week'] = st.number_input("Enter hours per week:", value=0)

    # Convert categorical values to numerical using LabelEncoder
    lb = LabelEncoder()
    encoded_inputs = {col: lb.fit_transform([feature_inputs[col]])[0] for col in feature_inputs}

    # Create a "Predict" button
    if st.button("Predict"):
        # Convert user inputs to a DataFrame
        user_inputs = pd.DataFrame.from_dict(feature_inputs, orient='index').T

        # # Make predictions based on user inputs
        # predicted_class = loaded_model.predict(user_inputs)
        # # Display the predicted class
        # st.write("Predicted Income Class:", predicted_class[0])


        # Make predictions based on user inputs for each model
        for model_name, loaded_model in loaded_models.items():
            predicted_class = loaded_model.predict(user_inputs)
            st.write(f"Predicted Income Class ({model_name}):", predicted_class[0])




def documentation_page():
    st.subheader("Documentation Page")
    # Add your documentation content here

    st.title("Graph Comparison in Streamlit")

    # Sidebar for graph selection
    graph_choice = st.sidebar.selectbox("Select Graph", ["Training and Testing Scores", "Percentage of Misclassification", "Accuracy"])

    # Scores and metrics for each mode
    modes = ['Decession Tree', 'Random forest', 'KNN', 'Logistic Regression']
    training_scores = [0.8230958230958231, 0.8263206388206388, 0.8113098894348895, 0.7602119164619164]
    testing_scores = [0.8072788697788698, 0.816953316953317, 0.7847051597051597, 0.7618243243243243]

    # Percentage of misclassification data
    misclassification = [0.19272113022113024, 0.18304668304668303, 0.21529484029484025, 0.23817567567567566]

    # Accuracy data
    accuracy = [1 - mis for mis in misclassification]

    if graph_choice == "Training and Testing Scores":
        # Plot training and testing scores graph
        fig_scores = plt.figure(figsize=(10, 6))
        bar_width = 0.4
        plt.bar([i - bar_width/2 for i in range(len(modes))], training_scores, width=bar_width, label='Training Score', color='blue')
        plt.bar([i + bar_width/2 for i in range(len(modes))], testing_scores, width=bar_width, label='Testing Score', color='orange', alpha=0.7)

        # Display values on top of each column
        for i, val in enumerate(training_scores):
            plt.text(i - bar_width/2, val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        for i, val in enumerate(testing_scores):
            plt.text(i + bar_width/2, val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Modes', fontsize=14)
        plt.ylabel('Scores', fontsize=14)
        plt.title('Training and Testing Scores Comparison', fontsize=16)
        plt.xticks(range(len(modes)), modes)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Display the training and testing scores graph in the Streamlit app
        st.subheader("Training and Testing Scores")
        st.pyplot(fig_scores)  # Pass the Matplotlib figure as an argument

    elif graph_choice == "Percentage of Misclassification":
        # Plot percentage of misclassification graph
        fig_misclassification = plt.figure(figsize=(10, 6))
        plt.bar(modes, misclassification, color='red')

        # Display values on top of each column
        for i, val in enumerate(misclassification):
            plt.text(i, val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Modes', fontsize=14)
        plt.ylabel('Percentage of Misclassification', fontsize=14)
        plt.title('Percentage of Misclassification Comparison', fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        # Display the percentage of misclassification graph in the Streamlit app
        st.subheader("Percentage of Misclassification")
        st.pyplot(fig_misclassification)  # Pass the Matplotlib figure as an argument

    elif graph_choice == "Accuracy":
        # Plot accuracy graph
        fig_accuracy = plt.figure(figsize=(10, 6))
        plt.bar(modes, accuracy, color='green')

        # Display values on top of each column
        for i, val in enumerate(accuracy):
            plt.text(i, val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Modes', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Accuracy Comparison', fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        # Display the accuracy graph in the Streamlit app
        st.subheader("Accuracy")
        st.pyplot(fig_accuracy)  # Pass the Matplotlib figure as an argument





if __name__ == "__main__":
    main()


# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()

# # # Load the saved model using pickle
# # model_filename = 'decision_tree_model.pkl'
# # with open(model_filename, 'rb') as model_file:
# #     loaded_model = pickle.load(model_file)

# # Load the saved models using pickle
# model_filenames = {
#     'Decision Tree': 'decision_tree_model.pkl',
#     'KNN': 'knn.pkl',
#     'Logistic Regression': 'Logistic_Regression.pkl',
#     'Random Forest': 'random_forest_model.pkl'
# }

# loaded_models = {model_name: pickle.load(open(filename, 'rb')) for model_name, filename in model_filenames.items()}

# # Load the unique values for categorical columns
# categorical_values = {
#     'Workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'unknown'],
#     'Education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
#     'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'unknown'],
#     'sex': ['Male', 'Female']
# }

# # Streamlit app
# st.title("Income Prediction App")

# # User input for features
# feature_inputs = {}
# feature_inputs['Age'] = st.number_input("Enter Age:", value=0)

# for col in ['Workclass', 'Education', 'occupation', 'sex']:
#     selected_option = st.selectbox(f"Select {col}:", categorical_values[col])
#     feature_inputs[col] = selected_option
#     encoder.fit(categorical_values[col])  # Fit the encoder on the unique values
#     encoded_value = encoder.transform([selected_option])[0]
#     feature_inputs[col] = encoded_value

# feature_inputs['hours_per_week'] = st.number_input("Enter hours per week:", value=0)

# # Convert categorical values to numerical using LabelEncoder
# lb = LabelEncoder()
# encoded_inputs = {col: lb.fit_transform([feature_inputs[col]])[0] for col in feature_inputs}

# # Create a "Predict" button
# if st.button("Predict"):
#     # Convert user inputs to a DataFrame
#     user_inputs = pd.DataFrame.from_dict(feature_inputs, orient='index').T

#     # # Make predictions based on user inputs
#     # predicted_class = loaded_model.predict(user_inputs)
#     # # Display the predicted class
#     # st.write("Predicted Income Class:", predicted_class[0])


#     # Make predictions based on user inputs for each model
#     for model_name, loaded_model in loaded_models.items():
#         predicted_class = loaded_model.predict(user_inputs)
#         st.write(f"Predicted Income Class ({model_name}):", predicted_class[0])


# def main():
#     st.title("Graph Comparison in Streamlit")

#     # Sidebar for graph selection
#     graph_choice = st.sidebar.selectbox("Select Graph", ["Training and Testing Scores", "Percentage of Misclassification", "Accuracy"])

#     # Scores and metrics for each mode
#     modes = ['Decession Tree', 'Random forest', 'KNN', 'Logistic Regression']
#     training_scores = [0.8230958230958231, 0.8263206388206388, 0.8113098894348895, 0.7602119164619164]
#     testing_scores = [0.8072788697788698, 0.816953316953317, 0.7847051597051597, 0.7618243243243243]

#     # Percentage of misclassification data
#     misclassification = [0.19272113022113024, 0.18304668304668303, 0.21529484029484025, 0.23817567567567566]

#     # Accuracy data
#     accuracy = [1 - mis for mis in misclassification]

#     if graph_choice == "Training and Testing Scores":
#         # Plot training and testing scores graph
#         fig_scores = plt.figure(figsize=(10, 6))
#         bar_width = 0.4
#         plt.bar([i - bar_width/2 for i in range(len(modes))], training_scores, width=bar_width, label='Training Score', color='blue')
#         plt.bar([i + bar_width/2 for i in range(len(modes))], testing_scores, width=bar_width, label='Testing Score', color='orange', alpha=0.7)

#         # Display values on top of each column
#         for i, val in enumerate(training_scores):
#             plt.text(i - bar_width/2, val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
#         for i, val in enumerate(testing_scores):
#             plt.text(i + bar_width/2, val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

#         plt.xlabel('Modes', fontsize=14)
#         plt.ylabel('Scores', fontsize=14)
#         plt.title('Training and Testing Scores Comparison', fontsize=16)
#         plt.xticks(range(len(modes)), modes)
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()

#         # Display the training and testing scores graph in the Streamlit app
#         st.subheader("Training and Testing Scores")
#         st.pyplot(fig_scores)  # Pass the Matplotlib figure as an argument

#     elif graph_choice == "Percentage of Misclassification":
#         # Plot percentage of misclassification graph
#         fig_misclassification = plt.figure(figsize=(10, 6))
#         plt.bar(modes, misclassification, color='red')

#         # Display values on top of each column
#         for i, val in enumerate(misclassification):
#             plt.text(i, val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

#         plt.xlabel('Modes', fontsize=14)
#         plt.ylabel('Percentage of Misclassification', fontsize=14)
#         plt.title('Percentage of Misclassification Comparison', fontsize=16)
#         plt.grid(True)
#         plt.tight_layout()

#         # Display the percentage of misclassification graph in the Streamlit app
#         st.subheader("Percentage of Misclassification")
#         st.pyplot(fig_misclassification)  # Pass the Matplotlib figure as an argument

#     elif graph_choice == "Accuracy":
#         # Plot accuracy graph
#         fig_accuracy = plt.figure(figsize=(10, 6))
#         plt.bar(modes, accuracy, color='green')

#         # Display values on top of each column
#         for i, val in enumerate(accuracy):
#             plt.text(i, val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

#         plt.xlabel('Modes', fontsize=14)
#         plt.ylabel('Accuracy', fontsize=14)
#         plt.title('Accuracy Comparison', fontsize=16)
#         plt.grid(True)
#         plt.tight_layout()

#         # Display the accuracy graph in the Streamlit app
#         st.subheader("Accuracy")
#         st.pyplot(fig_accuracy)  # Pass the Matplotlib figure as an argument

# if __name__ == "__main__":
#     main()
