from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.stats import norm

app = Flask(__name__)
CORS(app)

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes Call Option Price
    S: Stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def bond_valuation(face_value, coupon_rate, years_to_maturity, market_rate):
    """
    Calculate Bond Price and Yield
    face_value: Par value of the bond
    coupon_rate: Annual coupon rate
    years_to_maturity: Time until bond matures
    market_rate: Current market interest rate
    """
    coupon_payment = face_value * coupon_rate
    
    # Present value of coupon payments
    coupon_pv = coupon_payment * ((1 - (1 + market_rate)**-years_to_maturity) / market_rate)
    
    # Present value of face value
    face_value_pv = face_value / (1 + market_rate)**years_to_maturity
    
    bond_price = coupon_pv + face_value_pv
    return bond_price

def stock_valuation(dividends, growth_rate, required_return):
    """
    Calculate Stock Value using Dividend Discount Model
    dividends: Current annual dividend
    growth_rate: Expected dividend growth rate
    required_return: Investor's required rate of return
    """
    if growth_rate >= required_return:
        return "Invalid: Growth rate must be less than required return"
    
    stock_value = dividends / (required_return - growth_rate)
    return stock_value

def calculate_portfolio_volatility(weights, volatilities, correlations):
    """
    Calculate Portfolio Volatility using correlation matrix
    weights: List of asset weights
    volatilities: List of individual asset volatilities
    correlations: Correlation matrix as a list of lists
    """
    n = len(weights)
    if n != len(volatilities) or n != len(correlations) or any(len(row) != n for row in correlations):
        raise ValueError("Dimensions mismatch in input parameters")
    
    # Convert to numpy arrays
    weights = np.array(weights)
    volatilities = np.array(volatilities)
    correlations = np.array(correlations)
    
    # Create covariance matrix
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i,j] = correlations[i,j] * volatilities[i] * volatilities[j]
    
    # Calculate portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Return portfolio volatility (standard deviation)
    return np.sqrt(portfolio_variance)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_option', methods=['POST'])
def calculate_option():
    data = request.json
    try:
        price = black_scholes_call(
            float(data['stock_price']),
            float(data['strike_price']),
            float(data['time_to_maturity']),
            float(data['risk_free_rate']),
            float(data['volatility'])
        )
        return jsonify({'option_price': round(price, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calculate_bond', methods=['POST'])
def calculate_bond():
    data = request.json
    try:
        price = bond_valuation(
            float(data['face_value']),
            float(data['coupon_rate']),
            float(data['years_to_maturity']),
            float(data['market_rate'])
        )
        return jsonify({'bond_price': round(price, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calculate_stock', methods=['POST'])
def calculate_stock():
    data = request.json
    try:
        value = stock_valuation(
            float(data['dividends']),
            float(data['growth_rate']),
            float(data['required_return'])
        )
        return jsonify({'stock_value': round(value, 2) if isinstance(value, float) else value})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calculate_portfolio_volatility', methods=['POST'])
def portfolio_volatility():
    data = request.json
    try:
        weights = data['weights']
        volatilities = data['volatilities']
        correlations = data['correlations']
        
        # Convert percentages to decimals
        weights = [float(w)/100 for w in weights]
        volatilities = [float(v)/100 for v in volatilities]
        correlations = [[float(c) for c in row] for row in correlations]
        
        # Validate weights sum to 1
        if abs(sum(weights) - 1.0) > 0.0001:
            return jsonify({'error': 'Weights must sum to 100%'}), 400
            
        portfolio_vol = calculate_portfolio_volatility(weights, volatilities, correlations)
        return jsonify({'portfolio_volatility': round(portfolio_vol * 100, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
