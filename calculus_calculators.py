import numpy as np
from scipy import integrate
from sympy import symbols, diff, integrate as sym_integrate, sympify, lambdify

def calculate_derivative(expression, variable='x', point=None):
    """Calculate the derivative of a mathematical expression."""
    try:
        x = symbols(variable)
        expr = sympify(expression)
        derivative = diff(expr, x)
        
        if point is not None:
            # Evaluate at specific point
            result = derivative.subs(x, float(point))
            return {
                'derivative': str(derivative),
                'point_value': float(result)
            }
        return {
            'derivative': str(derivative)
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_integral(expression, variable='x', lower_bound=None, upper_bound=None):
    """Calculate the integral of a mathematical expression."""
    try:
        x = symbols(variable)
        expr = sympify(expression)
        
        if lower_bound is not None and upper_bound is not None:
            # Definite integral
            result = sym_integrate(expr, (x, lower_bound, upper_bound))
            return {
                'integral': str(sym_integrate(expr, x)),
                'definite_integral': float(result)
            }
        else:
            # Indefinite integral
            result = sym_integrate(expr, x)
            return {
                'integral': str(result)
            }
    except Exception as e:
        return {'error': str(e)}

def calculate_limit(expression, variable='x', point=None, direction=None):
    """Calculate the limit of a mathematical expression."""
    try:
        x = symbols(variable)
        expr = sympify(expression)
        
        if direction == 'left':
            point_value = expr.limit(x, point, '-')
        elif direction == 'right':
            point_value = expr.limit(x, point, '+')
        else:
            point_value = expr.limit(x, point)
            
        return {
            'limit': str(point_value)
        }
    except Exception as e:
        return {'error': str(e)}
