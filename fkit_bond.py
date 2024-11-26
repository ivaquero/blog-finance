import itertools
import math

import numpy as np
from scipy import optimize
from scipy import stats


def zero_coupon_bond(face_val, year_yield, time):
    """
    Price a zero coupon bond.

    face_val: face value of the bond.
    year_yield: annual yield or rate of the bond.
    time: time to maturity, in years.
    """
    return face_val / (1 + year_yield) ** time


class BootstrapYieldCurve:
    def __init__(self):
        self.zero_rates = {}
        self.instruments = {}

    def add_instrument(self, par, T, coup, price, compounding_freq=2):
        self.instruments[T] = (par, coup, price, compounding_freq)

    def get_maturities(self):
        """
        :return: a list of maturities of added instruments
        """
        return sorted(self.instruments.keys())

    def get_zero_rates(self):
        """
        Returns a list of spot rates on the yield curve.
        """
        self.bootstrap_zero_coupons()
        self.get_bond_spot_rates()
        return [self.zero_rates[T] for T in self.get_maturities()]

    def bootstrap_zero_coupons(self):
        """
        Bootstrap the yield curve with zero coupon instruments firstats.
        """
        for T, instrument in self.instruments.items():
            (par, coup, price, _) = instrument
            if coup == 0:
                spot_rate = self.zero_coupon_spot_rate(par, price, T)
                self.zero_rates[T] = spot_rate

    def zero_coupon_spot_rate(self, par, price, T):
        """
        :return: the zero coupon spot rate with continuous compounding.
        """
        return math.log(par / price) / T

    def get_bond_spot_rates(self):
        """
        Get spot rates implied by bonds, using short-term instruments.
        """
        for T in self.get_maturities():
            instrument = self.instruments[T]
            (_, coup, _, _) = instrument
            if coup != 0:
                spot_rate = self.calculate_bond_spot_rate(T, instrument)
                self.zero_rates[T] = spot_rate

    def calculate_bond_spot_rate(self, T, instrument):
        try:
            (par, coup, price, freq) = instrument
            periods = T * freq
            value = price
            per_coupon = coup / freq
            for i in range(int(periods) - 1):
                t = (i + 1) / float(freq)
                spot_rate = self.zero_rates[t]
                discounted_coupon = per_coupon * math.exp(-spot_rate * t)
                value -= discounted_coupon

            last_period = int(periods) / float(freq)
            return -math.log(value / (par + per_coupon)) / last_period
        except KeyError as e:
            print(f"Error: spot rate not found for T={t}. Missing key: {e}")


def bond_ytm(price, face_val, T, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = T * 2
    coupon = coup / 100.0 * face_val
    dt = [(i + 1) / freq for i in range(int(periods))]

    def ytm_func(y):
        return (
            sum(coupon / freq / (1 + y / freq) ** (freq * t) for t in dt)
            + face_val / (1 + y / freq) ** (freq * T)
            - price
        )

    return optimize.newton(ytm_func, guess)


def bond_price(face_val, T, ytm, coup, freq=2):
    freq = float(freq)
    periods = T * 2
    coupon = coup / 100.0 * face_val
    dt = [(i + 1) / freq for i in range(int(periods))]
    return sum(
        coupon / freq / (1 + ytm / freq) ** (freq * t) for t in dt
    ) + face_val / (1 + ytm / freq) ** (freq * T)


def bond_mod_duration(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)

    ytm_minus = ytm - dy
    price_minus = bond_price(par, T, ytm_minus, coup, freq)

    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)

    return (price_minus - price_plus) / (2 * price * dy)


def bond_convexity(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)

    ytm_minus = ytm - dy
    price_minus = bond_price(par, T, ytm_minus, coup, freq)

    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)

    return (price_minus + price_plus - 2 * price) / (price * dy**2)


def vasicek(r0, K, theta, sigma, T=1.0, N=10, seed=777):
    np.random.seed(seed)
    dt = T / float(N)
    rates = [r0]
    for _i in range(N):
        dr = K * (theta - rates[-1]) * dt + sigma * math.sqrt(dt) * np.random.normal()
        rates.append(rates[-1] + dr)

    return range(N + 1), rates


def CIR(r0, K, theta, sigma, T=1.0, N=10, seed=777):
    np.random.seed(seed)
    dt = T / float(N)
    rates = [r0]
    for _i in range(N):
        dr = (
            K * (theta - rates[-1]) * dt
            + sigma * math.sqrt(rates[-1]) * math.sqrt(dt) * np.random.normal()
        )
        rates.append(rates[-1] + dr)

    return range(N + 1), rates


def rendleman_bartter(r0, theta, sigma, T=1.0, N=10, seed=777):
    np.random.seed(seed)
    dt = T / float(N)
    rates = [r0]
    for _i in range(N):
        dr = (
            theta * rates[-1] * dt
            + sigma * rates[-1] * math.sqrt(dt) * np.random.normal()
        )
        rates.append(rates[-1] + dr)

    return range(N + 1), rates


def brennan_schwartz(r0, K, theta, sigma, T=1.0, N=10, seed=777):
    np.random.seed(seed)
    dt = T / float(N)
    rates = [r0]
    for _i in range(N):
        dr = (
            K * (theta - rates[-1]) * dt
            + sigma * rates[-1] * math.sqrt(dt) * np.random.normal()
        )
        rates.append(rates[-1] + dr)

    return range(N + 1), rates


class ForwardRates:
    def __init__(self):
        self.forward_rates = []
        self.spot_rates = {}

    def add_spot_rate(self, T, spot_rate):
        self.spot_rates[T] = spot_rate

    def get_forward_rates(self):
        """
        Returns a list of forward rates
        starting from the second time period.
        """
        periods = sorted(self.spot_rates.keys())
        for T2, T1 in itertools.pairwise(periods):
            forward_rate = self.calculate_forward_rate(T1, T2)
            self.forward_rates.append(forward_rate)

        return self.forward_rates

    def calculate_forward_rate(self, T1, T2):
        R1 = self.spot_rates[T1]
        R2 = self.spot_rates[T2]
        return (R2 * T2 - R1 * T1) / (T2 - T1)


class VasicekCZCB:
    def __init__(self):
        self.norminv = stats.distributions.norm.ppf
        self.norm = stats.distributions.norm.cdf

    def vasicek_czcb_values(
        self,
        r0,
        R,
        ratio,
        T,
        sigma,
        kappa,
        theta,
        M,
        prob=1e-6,
        max_policy_iter=10,
        grid_struct_const=0.25,
        rs=None,
    ):
        (r_min, dr, N, dtau) = self.vasicek_params(
            r0, M, sigma, kappa, theta, T, prob, grid_struct_const, rs
        )
        r = np.r_[0:N] * dr + r_min
        v_mplus1 = np.ones(N)

        for i in range(1, M + 1):
            K = self.exercise_call_price(R, ratio, i * dtau)
            eex = np.ones(N) * K
            (subdiagonal, diagonal, superdiagonal) = self.vasicek_diagonals(
                sigma, kappa, theta, r_min, dr, N, dtau
            )
            (v_mplus1, _iterations) = self.iterate(
                subdiagonal, diagonal, superdiagonal, v_mplus1, eex, max_policy_iter
            )
        return r, v_mplus1

    def vasicek_params(
        self, r0, M, sigma, kappa, theta, T, prob, grid_struct_const=0.25, rs=None
    ):
        if rs is not None:
            (r_min, r_max) = (rs[0], rs[-1])
        else:
            (r_min, r_max) = self.vasicek_limits(r0, sigma, kappa, theta, T, prob)

        dt = T / float(M)
        N = self.calculate_N(grid_struct_const, dt, sigma, r_max, r_min)
        dr = (r_max - r_min) / (N - 1)

        return (r_min, dr, N, dt)

    def calculate_N(self, max_structure_const, dt, sigma, r_max, r_min):
        N = 0
        while True:
            N += 1
            grid_structure_interval = (
                dt * (sigma**2) / (((r_max - r_min) / float(N)) ** 2)
            )
            if grid_structure_interval > max_structure_const:
                break
        return N

    def vasicek_limits(self, r0, sigma, kappa, theta, T, prob=1e-6):
        er = theta + (r0 - theta) * math.exp(-kappa * T)
        variance = (
            (sigma**2) * T
            if kappa == 0
            else (sigma**2) / (2 * kappa) * (1 - math.exp(-2 * kappa * T))
        )
        stdev = math.sqrt(variance)
        r_min = self.norminv(prob, er, stdev)
        r_max = self.norminv(1 - prob, er, stdev)
        return (r_min, r_max)

    def vasicek_diagonals(self, sigma, kappa, theta, r_min, dr, N, dtau):
        rn = np.r_[0:N] * dr + r_min
        subdiagonals = kappa * (theta - rn) * dtau / (2 * dr) - 0.5 * (
            sigma**2
        ) * dtau / (dr**2)
        diagonals = 1 + rn * dtau + sigma**2 * dtau / (dr**2)
        superdiagonals = -kappa * (theta - rn) * dtau / (2 * dr) - 0.5 * (
            sigma**2
        ) * dtau / (dr**2)

        # Implement boundary conditions.
        if N > 0:
            v_subd0 = subdiagonals[0]
            superdiagonals[0] = superdiagonals[0] - subdiagonals[0]
            diagonals[0] += 2 * v_subd0
            subdiagonals[0] = 0

        if N > 1:
            v_superd_last = superdiagonals[-1]
            superdiagonals[-1] = superdiagonals[-1] - subdiagonals[-1]
            diagonals[-1] += 2 * v_superd_last
            superdiagonals[-1] = 0

        return (subdiagonals, diagonals, superdiagonals)

    def check_exercise(self, V, eex):
        return eex < V

    def exercise_call_price(self, R, ratio, tau):
        return ratio * np.exp(-R * tau)

    def vasicek_policy_diagonals(
        self, subdiagonal, diagonal, superdiagonal, v_old, v_new, eex
    ):
        has_early_exercise = self.check_exercise(v_new, eex)
        subdiagonal[has_early_exercise] = 0
        superdiagonal[has_early_exercise] = 0
        policy = v_old / eex
        policy_values = policy[has_early_exercise]
        diagonal[has_early_exercise] = policy_values
        return (subdiagonal, diagonal, superdiagonal)

    def iterate(
        self, subdiagonal, diagonal, superdiagonal, v_old, eex, max_policy_iter=10
    ):
        v_mplus1 = v_old
        v_m = v_old
        change = np.zeros(len(v_old))
        prev_changes = np.zeros(len(v_old))

        iterations = 0
        while iterations <= max_policy_iter:
            iterations += 1

            v_mplus1 = self.tridiagonal_solve(
                subdiagonal, diagonal, superdiagonal, v_old
            )
            subdiagonal, diagonal, superdiagonal = self.vasicek_policy_diagonals(
                subdiagonal, diagonal, superdiagonal, v_old, v_mplus1, eex
            )

            is_eex = self.check_exercise(v_mplus1, eex)
            change[is_eex] = 1

            if iterations > 1:
                change[v_mplus1 != v_m] = 1

            is_no_more_eex = True not in is_eex
            if is_no_more_eex:
                break

            v_mplus1[is_eex] = eex[is_eex]
            changes = change == prev_changes

            is_no_further_changes = all((x == 1) for x in changes)
            if is_no_further_changes:
                break

            prev_changes = change
            v_m = v_mplus1

        return v_mplus1, iterations - 1

    def tridiagonal_solve(self, a, b, c, d):
        nf = len(a)  # Number of equations
        ac, bc, cc, dc = map(np.array, (a, b, c, d))  # Copy the array
        for it in range(1, nf):
            mc = ac[it] / bc[it - 1]
            bc[it] = bc[it] - mc * cc[it - 1]
            dc[it] = dc[it] - mc * dc[it - 1]

        xc = ac
        xc[-1] = dc[-1] / bc[-1]

        for il in range(nf - 2, -1, -1):
            xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

        del bc, cc, dc  # Delete variables from memory

        return xc
