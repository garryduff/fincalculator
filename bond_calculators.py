import numpy as np
from typing import Tuple, List, Dict

class Bond:
    """Base class for different types of bonds"""
    
    def __init__(self, face_value: float):
        self.face_value = face_value

    def get_price(self, interest_rate: float) -> float:
        """Calculate bond price given an interest rate"""
        raise NotImplementedError

    def get_yield_to_maturity(self, price: float) -> float:
        """Calculate yield to maturity given a price"""
        raise NotImplementedError

def future_value(present_value: float, interest_rate: float, years: int) -> float:
    """
    Calculate future value using compound interest formula.
    
    Args:
        present_value: Initial investment amount
        interest_rate: Annual interest rate (as decimal)
        years: Number of years
    
    Returns:
        Future value of the investment
    """
    return present_value * (1 + interest_rate) ** years

def present_value(future_value: float, interest_rate: float, years: int) -> float:
    """
    Calculate present value by discounting future value.
    
    Args:
        future_value: Future payment amount
        interest_rate: Annual interest rate (as decimal)
        years: Number of years
    
    Returns:
        Present value of the future payment
    """
    return future_value / (1 + interest_rate) ** years

class ZeroCouponBond(Bond):
    """Zero coupon bond that only pays face value at maturity"""
    
    def __init__(self, face_value: float, years_to_maturity: int):
        super().__init__(face_value)
        self.years_to_maturity = years_to_maturity
        
    def get_price(self, interest_rate: float) -> float:
        """
        Calculate zero coupon bond price.
        
        Args:
            interest_rate: Annual interest rate (as decimal)
            
        Returns:
            Current price of the bond
        """
        return present_value(self.face_value, interest_rate, self.years_to_maturity)
        
    def get_yield_to_maturity(self, price: float) -> float:
        """
        Calculate yield to maturity for a zero coupon bond.
        
        Args:
            price: Current market price of the bond
            
        Returns:
            Yield to maturity (as decimal)
        """
        return (self.face_value / price) ** (1 / self.years_to_maturity) - 1

class CouponBond(Bond):
    """Coupon bond that pays regular coupons and face value at maturity"""
    
    def __init__(self, face_value: float, coupon_rate: float, 
                 years_to_maturity: int, payments_per_year: int = 1):
        super().__init__(face_value)
        self.coupon_rate = coupon_rate
        self.years_to_maturity = years_to_maturity
        self.payments_per_year = payments_per_year
        self.coupon_payment = face_value * coupon_rate / payments_per_year
        
    def get_price(self, interest_rate: float) -> float:
        """
        Calculate coupon bond price.
        
        Args:
            interest_rate: Annual interest rate (as decimal)
            
        Returns:
            Current price of the bond
        """
        # Calculate PV of all coupon payments
        total_payments = self.years_to_maturity * self.payments_per_year
        coupon_pv = 0
        for t in range(1, total_payments + 1):
            years = t / self.payments_per_year
            coupon_pv += present_value(self.coupon_payment, interest_rate, years)
            
        # Add PV of final face value payment
        face_value_pv = present_value(self.face_value, interest_rate, 
                                    self.years_to_maturity)
        
        return coupon_pv + face_value_pv
        
    def get_yield_to_maturity(self, price: float, 
                             precision: float = 0.0001) -> float:
        """
        Calculate yield to maturity using linear interpolation.
        
        Args:
            price: Current market price of the bond
            precision: Desired precision for the yield calculation
            
        Returns:
            Yield to maturity (as decimal)
        """
        # Try some initial guesses for the yield
        r1 = 0.01  # 1%
        r2 = 0.15  # 15%
        
        p1 = self.get_price(r1)
        p2 = self.get_price(r2)
        
        # Use linear interpolation to get better estimate
        r = r1 + (price - p1)/(p2 - p1) * (r2 - r1)
        
        return r

class Perpetuity(Bond):
    """Perpetual bond that pays coupons forever with no maturity"""
    
    def __init__(self, coupon_payment: float):
        super().__init__(0)  # No face value for perpetuity
        self.coupon_payment = coupon_payment
        
    def get_price(self, interest_rate: float) -> float:
        """
        Calculate perpetuity price.
        
        Args:
            interest_rate: Annual interest rate (as decimal)
            
        Returns:
            Current price of the perpetuity
        """
        return self.coupon_payment / interest_rate
        
    def get_yield_to_maturity(self, price: float) -> float:
        """
        Calculate yield to maturity for perpetuity.
        
        Args:
            price: Current market price of the perpetuity
            
        Returns:
            Yield to maturity (as decimal)
        """
        return self.coupon_payment / price

def calculate_bond_price(face_value: float, coupon_rate: float, years_to_maturity: float,
                        yield_rate: float, payments_per_year: int = 1) -> float:
    """Calculate the price of a coupon bond given its characteristics and yield rate."""
    coupon_payment = face_value * coupon_rate / payments_per_year
    total_payments = int(years_to_maturity * payments_per_year)
    periodic_yield = yield_rate / payments_per_year
    
    coupon_pv = 0
    for t in range(1, total_payments + 1):
        coupon_pv += coupon_payment / (1 + periodic_yield) ** t
        
    face_value_pv = face_value / (1 + periodic_yield) ** total_payments
    
    return coupon_pv + face_value_pv

def find_yield_by_interpolation(face_value: float, coupon_rate: float, 
                              years_to_maturity: float, current_price: float,
                              payments_per_year: int = 1,
                              max_iterations: int = 10,
                              tolerance: float = 0.0001) -> float:
    """Find yield to maturity using linear interpolation method."""
    r_low = 0.0001
    r_high = 0.5
    
    for _ in range(max_iterations):
        p_low = calculate_bond_price(face_value, coupon_rate, years_to_maturity, 
                                   r_low, payments_per_year)
        p_high = calculate_bond_price(face_value, coupon_rate, years_to_maturity, 
                                    r_high, payments_per_year)
        
        r_new = r_low + (current_price - p_low) * (r_high - r_low) / (p_high - p_low)
        p_new = calculate_bond_price(face_value, coupon_rate, years_to_maturity, 
                                   r_new, payments_per_year)
        
        if abs(p_new - current_price) < tolerance:
            return r_new
            
        if p_new > current_price:
            r_low = r_new
        else:
            r_high = r_new
            
    return r_new

def calculate_duration(face_value: float, coupon_rate: float, years_to_maturity: float,
                      yield_rate: float, payments_per_year: int = 1) -> Dict[str, float]:
    """Calculate both Macaulay and Modified Duration for a bond."""
    coupon_payment = face_value * coupon_rate / payments_per_year
    periodic_yield = yield_rate / payments_per_year
    total_payments = int(years_to_maturity * payments_per_year)
    
    pv_sum = 0
    weighted_pv_sum = 0
    
    for t in range(1, total_payments + 1):
        time_years = t / payments_per_year
        pv = coupon_payment / (1 + periodic_yield) ** t
        pv_sum += pv
        weighted_pv_sum += pv * time_years
    
    pv_face = face_value / (1 + periodic_yield) ** total_payments
    pv_sum += pv_face
    weighted_pv_sum += pv_face * years_to_maturity
    
    macaulay_duration = weighted_pv_sum / pv_sum
    modified_duration = macaulay_duration / (1 + periodic_yield)
    
    return {
        "macaulay_duration": macaulay_duration,
        "modified_duration": modified_duration
    }

def estimate_price_change(duration: float, yield_change: float) -> float:
    """Estimate percentage price change using duration."""
    return -duration * yield_change * 100
