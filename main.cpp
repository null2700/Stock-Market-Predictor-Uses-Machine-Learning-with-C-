#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

using namespace mlpack;
using namespace mlpack::regression;
using namespace arma;
using namespace std;

int main() {
    
    mat data;
    data::Load("stock_data.csv", data, true);

    
    mat X = data.submat(0, 0, data.n_rows - 2, data.n_cols - 1);
    rowvec Y = data.row(data.n_rows - 1);

    
    LinearRegression lr(X, Y);

   
    rowvec predictions;
    lr.Predict(X, predictions);

    
    data::Save("predictions.csv", predictions);
    cout << "Stock Price Predictions Saved!" << endl;

    return 0;
}
