from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.stats import norm
from bond_calculators import (
    calculate_bond_price,
    find_yield_by_interpolation,
    calculate_duration,
    estimate_price_change
)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/options')
def options():
    return render_template('options.html')

@app.route('/bonds')
def bonds():
    return render_template('bonds.html')

@app.route('/stocks')
def stocks():
    return render_template('stocks.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/calculus')
def calculus():
    return render_template('calculus.html')

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

@app.route('/calculate_pv_fv', methods=['POST'])
def calculate_pv_fv():
    data = request.get_json()
    try:
        value = float(data['value'])
        interest_rate = float(data['interest_rate'])
        years = int(data['years'])
        calc_type = data['type']

        if calc_type == 'future':
            result = value * (1 + interest_rate) ** years
        else:  # present value
            result = value / ((1 + interest_rate) ** years)

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calculate_zero_coupon', methods=['POST'])
def calculate_zero_coupon():
    data = request.get_json()
    try:
        face_value = float(data['face_value'])
        years_to_maturity = int(data['years_to_maturity'])
        interest_rate = float(data['interest_rate'])

        # Calculate price
        price = face_value / ((1 + interest_rate) ** years_to_maturity)
        
        # Calculate YTM (same as interest rate for zero coupon)
        ytm = interest_rate

        return jsonify({
            'price': price,
            'ytm': ytm
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calculate_coupon_bond', methods=['POST'])
def calculate_coupon_bond():
    data = request.get_json()
    try:
        face_value = float(data['face_value'])
        coupon_rate = float(data['coupon_rate'])
        years_to_maturity = int(data['years_to_maturity'])
        payments_per_year = int(data['payments_per_year'])
        interest_rate = float(data['interest_rate'])

        # Calculate periodic values
        periodic_rate = interest_rate / payments_per_year
        total_periods = years_to_maturity * payments_per_year
        coupon_payment = (face_value * coupon_rate) / payments_per_year

        # Calculate present value of coupon payments
        pv_coupons = 0
        for t in range(1, total_periods + 1):
            pv_coupons += coupon_payment / ((1 + periodic_rate) ** t)

        # Calculate present value of face value
        pv_face_value = face_value / ((1 + periodic_rate) ** total_periods)

        # Total bond price
        price = pv_coupons + pv_face_value

        # Calculate YTM (simplified - using input rate)
        ytm = interest_rate

        return jsonify({
            'price': price,
            'ytm': ytm
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calculate_perpetuity', methods=['POST'])
def calculate_perpetuity():
    data = request.get_json()
    try:
        coupon_payment = float(data['coupon_payment'])
        interest_rate = float(data['interest_rate'])

        # Calculate perpetuity price
        price = coupon_payment / interest_rate

        # YTM is the same as interest rate
        ytm = interest_rate

        return jsonify({
            'price': price,
            'ytm': ytm
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calculate_yield', methods=['POST'])
def calculate_yield():
    data = request.get_json()
    try:
        face_value = float(data['face_value'])
        coupon_rate = float(data['coupon_rate'])
        years_to_maturity = float(data['years_to_maturity'])
        current_price = float(data['current_price'])
        payments_per_year = int(data.get('payments_per_year', 1))

        ytm = find_yield_by_interpolation(
            face_value=face_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            current_price=current_price,
            payments_per_year=payments_per_year
        )

        return jsonify({'yield': ytm})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calculate_duration', methods=['POST'])
def calculate_bond_duration():
    data = request.get_json()
    try:
        face_value = float(data['face_value'])
        coupon_rate = float(data['coupon_rate'])
        years_to_maturity = float(data['years_to_maturity'])
        yield_rate = float(data['yield_rate'])
        payments_per_year = int(data.get('payments_per_year', 1))

        durations = calculate_duration(
            face_value=face_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            yield_rate=yield_rate,
            payments_per_year=payments_per_year
        )

        return jsonify(durations)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/estimate_price_change', methods=['POST'])
def calculate_price_change():
    data = request.get_json()
    try:
        duration = float(data['duration'])
        yield_change = float(data['yield_change'])

        price_change = estimate_price_change(duration, yield_change)
        return jsonify({'price_change': price_change})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/calculate_derivative', methods=['POST'])
def calculate_derivative_route():
    data = request.json
    result = calculate_derivative(
        expression=data['expression'],
        point=data.get('point')
    )
    return jsonify(result)

@app.route('/calculate_integral', methods=['POST'])
def calculate_integral_route():
    data = request.json
    result = calculate_integral(
        expression=data['expression'],
        lower_bound=data.get('lower_bound'),
        upper_bound=data.get('upper_bound')
    )
    return jsonify(result)

@app.route('/calculate_limit', methods=['POST'])
def calculate_limit_route():
    data = request.json
    result = calculate_limit(
        expression=data['expression'],
        point=data.get('point'),
        direction=data.get('direction')
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
