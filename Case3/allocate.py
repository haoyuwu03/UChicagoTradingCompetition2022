import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, NonlinearConstraint

#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################

prices = pd.read_csv("Acutal Testing Data.csv")
analyst_1 = pd.read_csv("Predicted Testing Data Analyst 1.csv")
analyst_2 = pd.read_csv("Predicted Testing Data Analyst 2.csv")
analyst_3 = pd.read_csv("Predicted Testing Data Analyst 3.csv")
shares = pd.read_csv("Shares Outstanding.csv")
shares = shares[shares.columns[1:]]
prices = prices[prices.columns[1:]]
analyst_1 = analyst_1[analyst_1.columns[1:]]
analyst_2 = analyst_2[analyst_2.columns[1:]]
analyst_3 = analyst_3[analyst_3.columns[1:]]

def allocate_portfolio(asset_prices, asset_1,asset_2,asset_3):
    
    global prices, analyst_1,analyst_2,analyst_3,shares
    
    asset_1_np = np.array(asset_1)
    asset_2_np = np.array(asset_2)
    asset_3_np = np.array(asset_3)
    asset_np = np.array(asset_prices)
    
    expected_std, expected_ret = black_litter(prices, analyst_1, analyst_2, 
                                analyst_3,asset_1_np,asset_2_np,
                                asset_3_np, asset_np, shares)
    
    prices = prices.append(pd.Series(index = ["A","B","C","D","E","F","G","H","I"], data = asset_np), ignore_index = True)
    analyst_1 = analyst_1.append(pd.Series(index = ["A","B","C","D","E","F","G","H","I"], data = asset_1_np), ignore_index = True)
    analyst_2 = analyst_2.append(pd.Series(index = ["A","B","C","D","E","F","G","H","I"], data = asset_2_np), ignore_index = True)
    analyst_3  = analyst_3.append(pd.Series(index = ["A","B","C","D","E","F","G","H","I"], data = asset_3_np), ignore_index = True)
    
    initial_weights = np.repeat(1/9, 9)
    cons = NonlinearConstraint(constraint, 1, 1)
    res = minimize(generate_sharpe_ratio, 
                            initial_weights, 
                            args = (expected_ret, expected_std),
                            constraints = cons)
    
    return res.x

def constraint(x):
    return np.sum(np.abs(x))

def generate_ret(df):
    return np.log(df/df.shift(1))

def generate_ret_s(prev, curr):
    return np.log(curr/prev)

def generate_monthly_ret(df):
    mat = generate_ret(df)
    return mat.groupby(mat.index // 21).sum()

def calculate_sigma(df, daily = True):
    if(daily):
        return generate_ret(df).cov()
    return generate_monthly_ret(df).cov()

def calculate_pi(sh_out, sigma, l):
    _w = sh_out/np.sum(sh_out, axis = 1)[0]
    return l * np.matmul(sigma, _w.T)

# omega function
#   input: analyst_predictions, actual_prices, FUTURE ADDITION: weights for each month (default is all equal weighting over time, later maybe use sigmoid)
#   output: diagonal matrix with analyst variance for each stock 

def calculate_omega(analyst_preds, actual_prices):
    # get monthly returns, drop first month
    analyst_preds = analyst_preds[::21]
    anal_ret = generate_ret(analyst_preds).drop(0)
    actual_ret = generate_monthly_ret(actual_prices).drop(0)

    # Make difference df between each analyst's predicted returns and actual returns
    dif_rets = pd.DataFrame(anal_ret.values-actual_ret.values,columns=actual_ret.columns)

    # square each difference
    dif_rets_sq = dif_rets.pow(2)

    # FUTURE ADDITION: (weighted) average of the difference for each stock (each col)
    # start with just average
    avg_var = dif_rets_sq.mean(axis=0)

    # make diagonal matrix using the variance for each stock
    omega = pd.DataFrame(0,index=avg_var.index,columns=avg_var.index, dtype=avg_var.dtype)
    np.fill_diagonal(omega.values, avg_var)
    return omega

def gen_anal_weights(omega1, omega2, omega3):
    omega1 = 1/np.diag(omega1)
    omega2 = 1/np.diag(omega2)
    omega3 = 1/np.diag(omega3)
    
    total = omega1 + omega2 + omega3
    df = pd.DataFrame()
    df["Analyst_1"] = omega1
    df["Analyst_2"] = omega2
    df["Analyst_3"] = omega3
    df.index = ["A", "B", "C", "E","D", "F", "G", "H", "I"]
    df = df/total[:, None]
    return df

def avg_omega_plus_weights(anal1, anal2, anal3, prices):
    omega1 = calculate_omega(anal1, prices)
    omega2 = calculate_omega(anal2, prices)
    omega3 = calculate_omega(anal3, prices)
    
    weights = gen_anal_weights(omega1, omega2, omega3)
    
    new_df = pd.DataFrame()
    new_df["Anal1"] = np.diag(omega1)
    new_df["Anal2"] = np.diag(omega2)
    new_df["Anal3"] = np.diag(omega3)
    
    matrix = np.diag(np.matmul(weights, new_df.T))
    matrix = pd.Series(matrix, index = ["A", "B", "C", "E","D", "F", "G", "H", "I"])

    omega = pd.DataFrame(0,index=matrix.index,columns=matrix.index, dtype=matrix)
    np.fill_diagonal(omega.values, matrix)
    omega = omega/21
    
    return omega, weights

def calculate_Q(input1, input2, input3, weights):
    new_df = pd.DataFrame()
    new_df["Anal1"] = input1
    new_df["Anal2"] = input2
    new_df["Anal3"] = input3
    matrix = np.diag(np.matmul(weights, new_df.T))
    matrix = pd.Series(matrix, index = ["A", "B", "C", "E","D", "F", "G", "H", "I"])
    
    return matrix

def black_litter(df, anal1, anal2, anal3, input1, input2, input3, actual, sh_out):
    #pi is implied expected returns
    #sigma is variance of returns
    #Q is column vector of expected returns
    #omega is analyst covariance matrix of returns
    
    sigma = calculate_sigma(df, daily=False).values

    # try 1, 2, 3 for risk aversion parameter
    pi = calculate_pi(sh_out,sigma, 2)
    
    omega, w = avg_omega_plus_weights(anal1, anal2, anal3, df)
    
    in1_w = generate_ret_s(input1, actual)
    in2_w = generate_ret_s(input2, actual)
    in3_w = generate_ret_s(input3, actual)
    
    Q = calculate_Q(in1_w, in2_w, in3_w, w)
    p1 = np.linalg.inv(np.linalg.inv(omega) + np.linalg.inv(sigma))
    p2 = np.matmul(np.linalg.inv(sigma), pi) + np.matmul(np.linalg.inv(omega), Q[:, np.newaxis])
    out = np.matmul(p1, p2)
    
    return p1, out 

def generate_sharpe_ratio(weights, returns, variance):
    std = np.matmul(np.matmul(weights.T, variance), weights) ** 0.5
    expected = np.dot(weights, returns)
    return expected/std