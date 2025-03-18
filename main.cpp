#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

using namespace mlpack;
using namespace mlpack::regression;
using namespace arma;
using namespace std;

int main() {
    // Load the dataset (CSV file with historical stock data)
    mat data;
    data::Load("stock_data.csv", data, true);

    // Split into features (X) and target (Y)
    mat X = data.submat(0, 0, data.n_rows - 2, data.n_cols - 1);
    rowvec Y = data.row(data.n_rows - 1);

    // Train a Linear Regression model
    LinearRegression lr(X, Y);

    // Predict next values (for testing, we use the same features)
    rowvec predictions;
    lr.Predict(X, predictions);

    // Save predictions to a file
    data::Save("predictions.csv", predictions);
    cout << "Stock Price Predictions Saved!" << endl;

    return 0;
}
