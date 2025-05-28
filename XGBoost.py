import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
import os
import xgboost as xgb
import matplotlib.pyplot as plt

def main():
    # load the data
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/atomic.csv'
    df = pd.read_csv(file_path)

    # convert the chemical formula to a composition object
    stc = StrToComposition()
    df = stc.featurize_dataframe(df, 'Element_Combination')

    # use ElementProperty to add descriptors
    ep = ElementProperty.from_preset("magpie")
    df = ep.featurize_dataframe(df, col_id='composition')

    # select features and target
    features = df.drop(['Element_Combination', 'composition', 'Lattice_Parameter', 'USFE'], axis=1)
    target = df[['USFE', 'Lattice_Parameter']]

    # Drop duplicates from the dataset
    df = df.drop_duplicates(subset=['Element_Combination'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features.loc[df.index], target.loc[df.index], test_size=0.20, random_state=42)

    # Convert the data into DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {
        'objective': 'reg:squarederror',  # Regression task
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'rmse'
    }
    num_round = 100
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    # Train the model with early stopping
    evals_result = {}
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10, evals_result=evals_result)

    # Make predictions
    y_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Calculate percent difference
    percent_diff_usfe = 100 * abs((y_test['USFE'] - y_pred[:, 0]) / y_test['USFE'])
    percent_diff_lattice = 100 * abs((y_test['Lattice_Parameter'] - y_pred[:, 1]) / y_test['Lattice_Parameter'])

    # Print the highest percent difference for both USFE and Lattice Parameter
    max_percent_diff_usfe = percent_diff_usfe.max()
    max_percent_diff_lattice = percent_diff_lattice.max()
    print(f"Highest Percent Difference for USFE: {max_percent_diff_usfe:.2f}%")
    print(f"Highest Percent Difference for Lattice Parameter: {max_percent_diff_lattice:.2f}%")

    # Calculate and print the average percent difference
    avg_percent_diff_usfe = percent_diff_usfe.mean()
    avg_percent_diff_lattice = percent_diff_lattice.mean()
    print(f"Average Percent Difference for USFE: {avg_percent_diff_usfe:.2f}%")
    print(f"Average Percent Difference for Lattice Parameter: {avg_percent_diff_lattice:.2f}%")

    # Save the test results to a CSV file
    results_df = pd.DataFrame(data={
        'Element_Combination': df.loc[X_test.index, 'Element_Combination'],
        'Actual_USFE': y_test['USFE'], 
        'Predicted_USFE': y_pred[:, 0],
        'Percent_Diff_USFE': percent_diff_usfe,
        'Actual_Lattice_Parameter': y_test['Lattice_Parameter'], 
        'Predicted_Lattice_Parameter': y_pred[:, 1],
        'Percent_Diff_Lattice': percent_diff_lattice
    })
    results_df.to_csv('test_results.csv', index=False)

    # Plot the evaluation results
    epochs = len(evals_result['train']['rmse'])
    x_axis = range(0, epochs)
    
    fig, ax = plt.subplots()
    ax.plot(x_axis[-60:], evals_result['train']['rmse'][-60:], label='Train')
    ax.plot(x_axis[-60:], evals_result['eval']['rmse'][-60:], label='Test')
    ax.legend()
    plt.ylabel('RMSE')
    plt.title('XGBoost RMSE (Last 60 Epochs)')
    plt.show()

if __name__ == '__main__':
    main()
