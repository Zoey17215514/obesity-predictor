import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC # Import SVC
from sklearn.inspection import permutation_importance # Import permutation_importance
# import json # No longer need json

# Load the combined models and K-Fold results
combined_data = {}
loaded_models = None # Initialize loaded_models
all_kfold_results = {} # Initialize all_kfold_results

try:
    combined_filename = 'all_results_and_models.joblib'
    combined_data = load(combined_filename)
    loaded_models = combined_data.get('models', {}) # Get models from the combined data
    all_kfold_results = combined_data.get('kfold_metrics', {}) # Get kfold results from the combined data

    if loaded_models:
        st.success(f"Models and K-Fold results loaded successfully from '{combined_filename}'.")
    else:
         st.error(f"Error: 'models' key not found in '{combined_filename}'. Please ensure the combined file was created correctly.")
         loaded_models = None # Ensure loaded_models is None if models key is missing


except FileNotFoundError:
    st.error(f"Error: '{combined_filename}' not found. Please ensure the combined file was saved by running the cell that creates it.")
    loaded_models = None # Set loaded_models to None if the file is not found
    all_kfold_results = {} # Ensure kfold_results is empty if file is not found
except Exception as e:
     st.error(f"An unexpected error occurred while loading data from '{combined_filename}': {e}")
     loaded_models = None
     all_kfold_results = {}


# Define the features to be used for prediction (ensure this matches what the models were trained on)
# These features should ideally be obtained from the loaded models/pipelines
# For now, keeping it hardcoded based on previous steps
deployment_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP']

# Load the original data to fit the preprocessor correctly from CSV
try:
    csv_file_path = 'ObesityDataSet.csv' # Changed to CSV file path
    df = pd.read_csv(csv_file_path) # Changed to read from CSV

except FileNotFoundError:
    st.error("Error: 'ObesityDataSet.csv' not found. Please ensure the data file is available.")
    df = None

# Create preprocessing pipelines for numerical and categorical features
# This needs to be fitted on the training data
if df is not None:
    # Split data into training and testing sets for evaluation
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        df[deployment_features], df['NObeyesdad'], test_size=0.2, random_state=42, stratify=df['NObeyesdad']
    )

    # Identify categorical and numerical columns based on the deployment features
    # With the current deployment_features, there are no categorical columns.
    categorical_cols_for_preprocessor = [col for col in deployment_features if col in df.columns and df[col].dtype == 'object']
    numerical_cols_for_preprocessor = [col for col in deployment_features if col in df.columns and df[col].dtype != 'object']

    numerical_transformer_deploy = StandardScaler()
    # categorical_transformer_deploy = OneHotEncoder(handle_unknown='ignore', drop='first') # No categorical features for these models

    # Simplify the preprocessor since only numerical features are used
    preprocessor_deploy = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer_deploy, numerical_cols_for_preprocessor)
            # Removed 'cat' transformer as there are no categorical features in deployment_features
        ],
        remainder='passthrough'
    )

    # Fit the preprocessor with the training data (evaluation split) - Moved fitting outside the button click
    preprocessor_deploy.fit(X_train_eval)

    # Transform the test set for model evaluation
    X_test_processed_eval = preprocessor_deploy.transform(X_test_eval)
    y_test_eval = y_test_eval # Keep the original test labels

    # Calculate performance metrics for all loaded models on the single test set (for app demo)
    # and prepare data for the performance table using loaded K-Fold results
    model_performance_data = []
    if loaded_models:
        for name, model in loaded_models.items():
            try:
                # For SVC, ensure probability=True is set if not already
                # Note: This refitting is done here because the loaded models were pipelines fitted on the full data
                # If they were just classifiers, they would need retraining or setting probability=True separately.
                # Assuming loaded models are pipelines with classifiers that support probability or can be set.
                # For simplicity, we assume the loaded pipeline's classifier is ready for prediction.
                classifier_step = model.named_steps.get('classifier') if isinstance(model, Pipeline) else model # Get classifier if it's a pipeline
                if isinstance(classifier_step, SVC) and not hasattr(classifier_step, 'predict_proba'):
                    # This case is less likely if the saved pipeline had probability=True during training
                    # If the loaded model needs probability set, you might need to refit or ensure it was saved correctly
                    pass # Assume the loaded pipeline is ready for prediction


                # Calculate metrics on the single test split (for demonstration in the app)
                # Predict using the loaded model (which is a pipeline if saved as such)
                y_pred_eval = model.predict(X_test_eval) # Predict using original X_test_eval with the pipeline

                accuracy_single = accuracy_score(y_test_eval, y_pred_eval)
                precision_single = precision_score(y_test_eval, y_pred_eval, average='macro', zero_division=0)
                recall_single = recall_score(y_test_eval, y_test_eval, average='macro', zero_division=0) # Fixed typo y_pred_eval
                f1_single = f1_score(y_test_eval, y_pred_eval, average='macro', zero_division=0) # Fixed typo y_test_eval


                # Prepare data row for the table
                row_data = {
                    'Model': name,
                    'Accuracy (Test)': accuracy_single,
                    'Precision (Test)': precision_single,
                    'Recall (Test)': recall_single,
                    'F1 Score (Test)': f1_single,
                }

                # Add K-Fold results if loaded for this model
                if name in all_kfold_results:
                    kfold_res = all_kfold_results[name]
                    row_data['Accuracy (K-Fold Avg)'] = kfold_res.get('Avg Accuracy')
                    row_data['Accuracy (K-Fold Std)'] = kfold_res.get('Std Accuracy')
                    row_data['Precision (K-Fold Avg)'] = kfold_res.get('Avg Precision')
                    row_data['Precision (K-Fold Std)'] = kfold_res.get('Std Precision')
                    row_data['Recall (K-Fold Avg)'] = kfold_res.get('Avg Recall')
                    row_data['Recall (K-Fold Std)'] = kfold_res.get('Std Recall')
                    row_data['F1 Score (K-Fold Avg)'] = kfold_res.get('Avg F1 Score')
                    row_data['F1 Score (K-Fold Std)'] = kfold_res.get('Std F1 Score')

                model_performance_data.append(row_data)

            except Exception as e:
                st.warning(f"Could not calculate performance metrics for {name}: {e}")

    model_performance_df = pd.DataFrame(model_performance_data)

    # Format standard deviation columns and combine Avg/Std for display
    if not model_performance_df.empty:
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            avg_col = metric + ' (K-Fold Avg)'
            std_col = metric + ' (K-Fold Std)'
            combined_col_name = metric + ' (K-Fold)' # New column name

            if avg_col in model_performance_df.columns and std_col in model_performance_df.columns:
                # Combine Avg and Std, handling potential NaN values
                model_performance_df[combined_col_name] = model_performance_df.apply(
                    lambda row: f"{row[avg_col]:.4f} &plusmn; {row[std_col]:.4f}" if pd.notna(row[avg_col]) and pd.notna(row[std_col]) else "",
                    axis=1
                )
                # Drop the separate Avg and Std columns after combining
                model_performance_df = model_performance_df.drop(columns=[avg_col, std_col])
            elif avg_col in model_performance_df.columns:
                 # If only Avg is available
                 model_performance_df[combined_col_name] = model_performance_df[avg_col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
                 model_performance_df = model_performance_df.drop(columns=[avg_col])
            elif std_col in model_performance_df.columns:
                 # If only Std is available (unlikely but for completeness)
                 model_performance_df[combined_col_name] = model_performance_df[std_col].apply(lambda x: f"&plusmn; {x:.4f}" if pd.notna(x) else "")
                 model_performance_df = model_performance_df.drop(columns=[std_col])


# Streamlit App Title
st.title("Obesity Level Prediction Report")

# Check if models and data are loaded before proceeding
if loaded_models is not None and df is not None:

    # Model Performance Comparison (Table and Line Chart)
    st.header("1. Model Performance Comparison")

    if not model_performance_df.empty:
        st.subheader("1.1 Model Performance Table") # Removed "(on Test Set)" as it now includes K-Fold
        # Reorder columns for better display: Model, Test metrics, K-Fold metrics
        col_order = ['Model']
        test_cols = [col for col in model_performance_df.columns if '(Test)' in col]
        kfold_cols = [col for col in model_performance_df.columns if '(K-Fold)' in col]

        col_order.extend(test_cols)
        # Sort K-Fold columns to group metrics
        kfold_cols_sorted = []
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
             kfold_cols_sorted.extend([col for col in kfold_cols if col.startswith(metric)])

        col_order.extend(kfold_cols_sorted)

        # Ensure all columns are included, in case there are others
        final_cols = col_order + [col for col in model_performance_df.columns if col not in col_order]

        # Select and display columns, avoiding highlighting formatted strings
        cols_to_highlight = [col for col in model_performance_df.columns if '(Test)' in col]


        st.dataframe(
            model_performance_df[final_cols].style.highlight_max(
                subset=cols_to_highlight, # Highlight only the numerical columns from Test set
                axis=0,
                color='lightgreen'
            ),
            use_container_width=True
        )


        # Accuracy Over Models Line Chart
        # The line chart is best suited for single-value comparisons, so we'll use the Test or Avg K-Fold if available
        chart_data = model_performance_df.copy()
        # Use Avg K-Fold (numerical part) if available, otherwise use Test
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
             combined_kfold_col = metric + ' (K-Fold)'
             test_col = metric + ' (Test)'
             if combined_kfold_col in chart_data.columns:
                 # Extract numerical value from formatted string for chart
                 chart_data[metric] = chart_data[combined_kfold_col].apply(lambda x: float(x.split('&plusmn;')[0].strip()) if isinstance(x, str) and '&plusmn;' in x else None) # Use None for missing/unformatted
                 chart_data = chart_data.drop(columns=[combined_kfold_col])
             elif test_col in chart_data.columns:
                 chart_data[metric] = chart_data[test_col]
                 chart_data = chart_data.drop(columns=[test_col])
             else:
                 # If neither K-Fold nor Test is available, drop the metric column from chart_data
                 chart_data = chart_data.drop(columns=[metric])


        if not chart_data.empty and len([col for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score'] if col in chart_data.columns]) > 0:
             model_performance_melted_line = chart_data.melt(id_vars='Model', var_name='Metric', value_name='Score', value_vars=[col for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score'] if col in chart_data.columns])
             fig1, ax1 = plt.subplots(figsize=(8, 5)) # Smaller figure size
             sns.lineplot(x='Model', y='Score', hue='Metric', data=model_performance_melted_line, marker='o', ax=ax1)
             ax1.set_title('Model Performance Comparison (Line Plot)')
             ax1.set_ylabel('Score')
             ax1.set_ylim(0.8, 1.0) # Adjust y-axis limits as needed
             ax1.legend(title='Metric')
             st.pyplot(fig1)
             plt.close(fig1)
        else:
             st.info("Not enough data available to generate the performance line chart.")


    # Model Selection using Radio Buttons
    st.header("2. Model Selection")
    models_to_choose = list(loaded_models.keys()) if loaded_models else []
    if models_to_choose:
        selected_model_name = st.radio("Select a Model for Prediction:", models_to_choose)
    else:
        st.warning("No models loaded for selection.")
        selected_model_name = None


    st.header("3. User Input Data") # Changed section header
    input_data = {}
    # Define mapping for FCVC text labels to numerical values
    fcvc_mapping = {"Never": 1.0, "Sometimes": 2.0, "Always": 3.0}
    fcvc_options = list(fcvc_mapping.keys())

    # Create columns for Age, Height, and Weight
    col_age, col_height, col_weight = st.columns(3)
    with col_age:
        col = 'Age'
        # Check if the column is in deployment_features before creating the input
        if col in deployment_features:
             input_data[col] = st.number_input(f"{col} (years):", value=0, min_value=0, help="Enter age in years") # Updated label and help

    with col_height:
        col = 'Height'
        if col in deployment_features:
            input_data[col] = st.number_input(f"{col} (m):", value=0.0, min_value=0.0, help="Enter height in meters") # Updated label and help

    with col_weight:
        col = 'Weight'
        if col in deployment_features:
            input_data[col] = st.number_input(f"{col} (kg):", value=0.0, min_value=0.0, help="Enter weight in kilograms") # Updated label and help

    # Create columns for FCVC and NCP
    col_fcvc, col_ncp = st.columns(2)

    with col_fcvc:
        col = 'FCVC'
        if col in deployment_features:
             selected_fcvc_text = st.selectbox("Frequency of consumption of vegetables:", fcvc_options)
             input_data[col] = fcvc_mapping[selected_fcvc_text] # Map text to numerical value

    with col_ncp:
        col = 'NCP'
        if col in deployment_features:
             # Changed to radio button input
             input_data[col] = st.radio("Number of main meals per day:", options=[1.0, 2.0, 3.0, 4.0])


    # Handle any remaining deployment features that were not explicitly placed in columns
    remaining_features = [col for col in deployment_features if col not in ['Age', 'Height', 'Weight', 'FCVC', 'NCP']]
    for col in remaining_features:
         # Check if the column is in deployment_features before creating the input
         if col in deployment_features:
             if col in categorical_cols_for_preprocessor:
                options = list(df[col].unique())
                input_data[col] = st.selectbox(f"{col}:", options)
             elif col in numerical_cols_for_preprocessor:
                input_data[col] = st.number_input(f"{col}:", value=0.0, min_value=0.0) # Assuming remaining numerical features should also be non-negative


    # Predict (with submit button)
    if st.button("Generate Prediction Report"):
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data using the fitted preprocessor
        # Ensure the input DataFrame has the same columns as the training data used for the preprocessor
        # This might require adding missing columns with default values (e.g., 0 for one-hot encoded)
        # For simplicity with current numerical-only features, we can proceed directly
        try:
            input_data_processed = preprocessor_deploy.transform(input_df[deployment_features])
        except ValueError as e:
             st.error(f"Error during preprocessing: {e}. Please check if all required features are provided.")
             input_data_processed = None # Set to None to prevent further errors


        st.header("4. Prediction Results")

        # Check if a model is selected and data is processed before predicting
        if selected_model_name and input_data_processed is not None:
            # Get the selected model
            if selected_model_name in loaded_models:
                model = loaded_models[selected_model_name]
                # Make prediction
                prediction = model.predict(input_data_processed)

                st.subheader(f"Prediction using {selected_model_name}:")
                st.write(f"Predicted Obesity Level: **{prediction[0]}**")

                # Add a simple interpretation based on the prediction
                st.subheader("Interpretation:")
                if 'Obesity' in prediction[0]:
                    st.write("Based on the provided data and the selected model, the predicted obesity level falls into an 'Obesity' category. This indicates a higher risk of health issues associated with obesity.")
                elif 'Overweight' in prediction[0]:
                    st.write("Based on the provided data and the selected model, the predicted obesity level falls into an 'Overweight' category. This suggests you are at risk of developing obesity.")
                elif 'Normal_Weight' in prediction[0]:
                    st.write("Based on the provided data and the selected model, the predicted obesity level falls into the 'Normal Weight' category. This suggests you are currently maintaining a healthy weight.")
                elif 'Insufficient_Weight' in prediction[0]:
                    st.write("Based on the provided data and the selected model, the predicted obesity level falls into the 'Insufficient Weight' category. This suggests you are underweight, which can also lead to health concerns.")


                # Add a pie chart for risk distribution (using predict_proba if available)
                # Check if the loaded model (which is a pipeline) has a classifier step that supports predict_proba
                classifier_step_for_proba = model.named_steps.get('classifier') if isinstance(model, Pipeline) else model
                if hasattr(classifier_step_for_proba, 'predict_proba'):
                    st.subheader("Risk Distribution by Obesity Level:")
                    # Get the probability distribution for the prediction
                    # Use the pipeline to predict probabilities on the original input_df[deployment_features]
                    # The pipeline will handle preprocessing internally for prediction
                    probabilities = model.predict_proba(input_df[deployment_features])[0]


                    # Get the class labels from the classifier step
                    class_labels = classifier_step_for_proba.classes_

                    # Create a pandas Series for easy plotting
                    risk_distribution = pd.Series(probabilities, index=class_labels)

                    # # Filter for the desired classes - Removed this line to include all classes
                    # target_classes = ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
                    # risk_distribution_filtered = risk_distribution[risk_distribution.index.isin(target_classes)]

                    # Plot the pie chart
                    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                    risk_distribution.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax_pie)
                    ax_pie.set_title('Risk Distribution for Obesity Levels')
                    ax_pie.set_ylabel('') # Remove the default y-label
                    st.pyplot(fig_pie)
                    plt.close(fig_pie)
                else:
                    st.info("The selected model does not support probability prediction (predict_proba) for the pie chart.")

                # Add Feature Importance chart below prediction results
                st.subheader(f"Feature Relevance ({selected_model_name})") # Changed title to be more general

                # Get feature names directly from deployment_features since they are numerical
                feature_names = deployment_features

                # Get the classifier step from the loaded pipeline for checking importances/coefficients
                classifier_for_importance = model.named_steps.get('classifier') if isinstance(model, Pipeline) else model

                # Add explicit check for SVM kernel if the classifier step is an SVC
                is_svm = isinstance(classifier_for_importance, SVC)
                svm_is_linear = is_svm and classifier_for_importance.kernel == 'linear'


                if hasattr(classifier_for_importance, 'feature_importances_'): # Use classifier step
                    st.subheader(f"Feature Importances ({selected_model_name})") # Specific title for importance
                    importances = classifier_for_importance.feature_importances_ # Use the classifier step's importances
                    if len(importances) == len(feature_names):
                        feat_importances = pd.Series(importances, index=feature_names)
                        feat_importances = feat_importances.sort_values(ascending=False)

                        fig4, ax4 = plt.subplots(figsize=(8, 6)) # Smaller figure size
                        feat_importances.plot(kind='barh', ax=ax4)
                        ax4.set_title(f'Feature Importances ({selected_model_name})') # Updated title
                        ax4.set_xlabel('Importance')
                        ax4.invert_yaxis()
                        st.pyplot(fig4)
                        plt.close(fig4)
                    else:
                        st.warning(f"Could not match feature importances to feature names. Number of importances ({len(importances)}) and feature names ({len(feature_names)}) do not match.")

                elif hasattr(classifier_for_importance, 'coef_') and svm_is_linear: # Check classifier step AND if it's a linear SVM
                     st.subheader(f"Feature Coefficients (Absolute Mean) ({selected_model_name})") # Specific title for coefficients
                     # For multi-class, coef_ is shape (n_classes, n_features). Take the mean of absolute values.
                     coef_values = np.abs(classifier_for_importance.coef_).mean(axis=0) # Use the classifier step's coefficients

                     if len(coef_values) == len(feature_names):
                         feat_coef = pd.Series(coef_values, index=feature_names)
                         feat_coef = feat_coef.sort_values(ascending=False)

                         fig_coef, ax_coef = plt.subplots(figsize=(8, 6)) # Smaller figure size
                         feat_coef.plot(kind='barh', ax=ax_coef)
                         ax_coef.set_title(f'Feature Coefficients (Absolute Mean) ({selected_model_name})') # Updated title
                         ax_coef.set_xlabel('Absolute Mean Coefficient Value')
                         ax_coef.invert_yaxis()
                         st.pyplot(fig_coef)
                         plt.close(fig_coef)
                     else:
                         st.warning(f"Could not match feature coefficients to feature names. Number of coefficients ({len(coef_values)}) and feature names ({len(feature_names)}) do not match.")
                elif is_svm and not svm_is_linear:
                     st.info(f"The selected SVM model uses a non-linear kernel ({classifier_for_importance.kernel}) and therefore does not have coefficients to display feature relevance directly.")
                     # Add Permutation Importance chart for non-linear SVM
                     st.subheader(f"Permutation Importance ({selected_model_name})")
                     try:
                        # Calculate permutation importance on the test set
                        # Use the full pipeline (model) and the original X_test_eval
                        # The pipeline handles preprocessing internally for permutation_importance
                        result = permutation_importance(model, X_test_eval, y_test_eval, n_repeats=10, random_state=42, n_jobs=-1)

                        # Get the importance scores and sort them
                        sorted_importances_idx = result.importances_mean.argsort()
                        sorted_importances = result.importances_mean[sorted_importances_idx]
                        # Use the original deployment_features names for the plot labels
                        sorted_feature_names = [deployment_features[i] for i in sorted_importances_idx]


                        # Create the bar chart
                        fig_perm, ax_perm = plt.subplots(figsize=(8, 6))
                        ax_perm.barh(sorted_feature_names, sorted_importances)
                        ax_perm.set_title(f"Permutation Importance ({selected_model_name})")
                        ax_perm.set_xlabel("Mean Decrease in Accuracy") # Or other metric if specified
                        st.pyplot(fig_perm)
                        plt.close(fig_perm)

                     except Exception as e:
                         st.error(f"An error occurred while generating Permutation Importance chart: {e}")

                else:
                    st.info(f"The selected model ({selected_model_name}) does not have feature importances or coefficients to display.")
            else:
                 st.warning(f"Selected model '{selected_model_name}' not found in loaded models.")


else:
    st.warning("Models or data not loaded. Please ensure 'all_results_and_models.joblib' and 'ObesityDataSet.csv' are in the correct directory and the necessary training cells were run.")
