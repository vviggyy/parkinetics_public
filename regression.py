
'''
Maps clinical scores associated with each patient identifier to rows in an output file 
of the pipeline.py (i.e. merges clinical scores with metrics files)

'''

import numpy as np 
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def parse_args():
    """Handles command-line arguments.

    Returns:
        args (a parser Namespace): Stores command line attributes.
    """
    parser = argparse.ArgumentParser(description="handles arguments for mapping")

    parser.add_argument(
        "-m",
        "--input",
        type=str,
        help="Path to combined metrics + clinical data file",
    )

    parser.add_argument(
        "-a",
        "--all",
        type = bool,
        help = "Attempt regression and correlation with all metrics and clinical scores",
        default = True
    )

    """"
    parser.add_argument(
        "-r",
        "--regression",
        type=bool,
        help="Attempt solely regression with all metrics and overall scores",
        default = False
    )
    parser.add_argument(
        "-c"
        "--correlation",
        type = bool,
        help = "Attempt correlation with all metrics and overall scores",
        default = False
    )
    )
    parser.add_argument(
        "-f",
        "--functional",
        type=bool,
        help="Regression with functional score",
        default = False
    )
    parser.add_argument(
        "-h",
        "--hand_used",
        type=bool,
        help="Regression only with hand_used",
        default = False
    )
    parser.add_argument(
        "-d",
        "--dominant_symptom",
        type=bool,
        help="Use the dominant symptom score on the hand of use",
        default=False
    )
    """
    
    return parser.parse_args()
def regression(input, output, des):
    X = input
    y = output
    # Split the data into training and testing sets (optional but recommended)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # R-squared value

    coefficients = model.coef_
    features = X.columns
    coefficients = [(coef, feature) for coef, feature in zip(coefficients, features) if abs(coef) > 1e-2]

    intercept = model.intercept_  # Intercept
    equation = f"Y = {intercept:.2f}"
    for item in coefficients:
        coef = item[0]
        feature_name = item[1]
        equation += f" + ({coef:.2f})*" + feature_name
        # Plotting
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')

    # Add text to show R^2 on the plot
    plt.text(0.5 * (y_test.min() + y_test.max()), 0.5 * (y_pred.min() + y_pred.max()), 
            f'R² = {r2:.2f}', color='black', fontsize=12, ha='center')

    # Labels and title
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title('Actual vs. Predicted Scores for' + des)

    # Show the plot
    plt.show()
    return [des, equation, r2, mse]

def main():
    """Main control flow.
    """
    args = parse_args() #get command-line attributes
    
    #read in both feature and clinical metrics...
    combined_features = pd.read_csv(args.input)
    model_results = []
    if args.all: #merge both
        # step 1 with all scores
        input = combined_features.loc[:,"skew" : "0_Zero crossing rate"]

        total_score_model = regression(input, combined_features["total_score"], "total score model")
        motor_score_model = regression(input, combined_features["total_score_18_31"], "motor score model")
        functional_score_model = regression(input, combined_features["total_functional_score"], "functional score model")
        hand_used_model = regression(input, combined_features["score_hand_used"], "hand_used specific model")
        model_results.extend([total_score_model, motor_score_model, functional_score_model, hand_used_model])

        dominance_A = combined_features[combined_features["Dominance"] == "A"]
        dominance_T = combined_features[combined_features["Dominance"] == "T"]  
        akinesia_specific_model = regression(dominance_A.loc[:,"skew" : "0_Zero crossing rate"], dominance_A["AKT_score"], "akinesia specific model")
        tremor_specific_model = regression(dominance_T.loc[:,"skew" : "0_Zero crossing rate"], dominance_T["tremor_score"], "tremor specific model")
        model_results.extend([akinesia_specific_model, tremor_specific_model])
        model_df = pd.DataFrame(model_results, columns=["Model", "Equation", "R-squared", "MSE"])
        print(model_df)

if __name__ == "__main__":
    main()
    