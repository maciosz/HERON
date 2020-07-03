#!/usr/bin/python

import copy
import logging

#from scipy.stats import nbinom
from scipy.special import digamma

import hmmlearn.mynumpy as np

# dl / dr = sum_t digamma(x_t + r) * post - sum_t digamma(r) * post + sum_t ln(p) * post
# dl / dr = sum_t post * (digamma(x_t + r) - digamma(r) + ln(p))

def calculate_derivative(posteriors, data, r, p):
    def _digamma(array):
        return digamma(array.astype("float64")).astype("float128")
    n_comp, n_var = r.shape
    n_obs, _ = data.shape

    derivative = np.zeros(r.shape)

    for state in range(n_comp):
        r_j, p_j = r[state], p[state]
        posteriors_j = posteriors[:, state][:, np.newaxis]
        in_brackets = _digamma(data + r_j) - _digamma(r_j) + np.log(p_j)
        derivative[state] = np.sum(posteriors_j * in_brackets, axis=0)

    #print("derivative:")
    #print(derivative)
    return derivative

    # it's the same, just different implementation
    n_comp, n_var = r.shape
    n_obs, _ = data.shape
    desired_shape = (n_comp, n_obs, n_var)
    data_repeated = np.concatenate([data] * n_comp).reshape(desired_shape)
    r_repeated = np.repeat([r], n_obs, axis=1).reshape(desired_shape)
    digamma_of_sum = _digamma(data_repeated + r_repeated)
    digamma_of_r = _digamma(r_repeated)
    log_p = np.log(p)
    log_p_repeated = np.repeat([log_p], n_obs, axis=1).reshape(desired_shape)
    sum_ = digamma_of_sum - digamma_of_r + log_p_repeated
    posteriors_repeated = np.repeat(posteriors.T, n_var).reshape(desired_shape)
    product = posteriors_repeated * sum_
    derivative = np.sum(product, axis=1)
    return derivative

    # another implementation
    r_conc = np.concatenate([r] * n_obs, axis=0)
    r_conc = r_conc.reshape(n_obs, n_comp, n_var)
    X_repeat = np.repeat(data, n_comp, axis=0)
    X_repeat = X_repeat.reshape(n_obs, n_comp, n_var)
    suma = X_repeat + r_conc
    suma = suma.reshape(n_obs, n_comp, n_var)
    pstwa_repeat = np.repeat(posteriors, n_var)
    pstwa_repeat = pstwa_repeat.reshape(n_obs, n_comp, n_var)
    a = np.sum(pstwa_repeat * _digamma(suma), axis=0)
    a = a.reshape(n_comp, n_var)
    b = np.sum(pstwa_repeat * _digamma(r_conc), axis=0)
    b = b.reshape(n_comp, n_var)
    p_conc = np.concatenate([p] * n_obs, axis=0)
    p_conc = p_conc.reshape(n_obs, n_comp, n_var)
    c = np.sum(pstwa_repeat * np.log(p_conc), axis=0)
    c = c.reshape(n_comp, n_var)
    derivative = a - b + c
    return derivative

def update_r(r, derivative, delta, stop):
    # mozna by jakos sprytniej skakac
    # np teraz mam ciag ktory skacze od kilkudziesieciu milionow
    #  na plusie i minusie (dwa razy minus, raz plus, i tak w kolko),
    #  zakres sie zaciesnia, ale powoli
    #  moze mozna by jakos uzaleznic od tego na ile duza co do modulu
    #  byla poprzednia i jeszcze poprzednia pochodna
    # albo jakos wazyc rejony czy co
    # najgorzej jak np jest caly czas ta sama dodatnia
    #  przeplatana dwoma ujemnymi, coraz blizszymi zera, ale wciaz odleglymi
    #  czlowiek by zobaczyl ze mozna tu skakac szybciej,
    #  zamiast ladowac w tym samym miejscu
    #  wiec powinno sie to tez dac zaimplementowac...
    # inna sprawa ze dla poczatkowych iteracji,
    #  kiedy estymacja p jest tez bardzo zgrubna,
    #  nie potrzebuje chyba bardzo precyzyjnej estymacji r;
    #  wystarczy mi nieduza pochodna, niekoniecznie bardzo bliska zero.
    #  Ale to juz bardziej skomplikowane do zaimplementowania.
    n_comp, n_var = r.shape
    for i in range(n_comp):
        for j in range(n_var):
            if stop[i, j] is True:
                continue
            if abs(derivative[i, j]) <= 1e-10:
                continue
            if delta[i, j] == 0:
                if derivative[i, j] < 0:
                    delta[i, j] = r[i, j] * -0.3
                elif derivative[i, j] > 0:
                    delta[i, j] = r[i, j] * 10 + 5
                else:
                    print("cos nie tak, pewnie nan")
            elif delta[i, j] < 0:
                if derivative[i, j] < 0:
                    pass
                elif derivative[i, j] > 0:
                    delta[i, j] *= -0.5
                else:
                    print("cos nie tak, pewnie nan")
            elif delta[i, j] > 0:
                if derivative[i, j] < 0:
                    delta[i, j] *= -0.5
                elif derivative[i, j] > 0:
                    pass
                else:
                    print("cos nie tak, pewnie nan")
            r[i, j] += delta[i, j]
            if r[i, j] <= 0:
                if derivative[i, j] < 0:
                    logging.warning("r mle < 0, derivative < 0")
                    #print("r bliskie zero, ale pochodna ujemna...")
                    #print("obczaj: %f %f %f" % (r[i, j], derivative[i, j], delta[i, j]))
                    stop[i, j] = True
                r[i, j] = 0.0001
                delta[i, j] = 0
    return r, delta, stop

def find_r(r_initial, dane, pstwa, p, threshold=1e-3):
    r = r_initial.copy()
    r_not_found = True
    counter = 0
    delta = np.zeros(shape=r.shape)
    stop = np.zeros(r.shape, dtype=bool)
    while r_not_found:
        derivative = calculate_derivative(pstwa, dane, copy.deepcopy(r), p)
        #print(derivative)
        if np.any(np.isnan(derivative)):
            print("Derivative is nan, stop this madness")
            print("That's the r, p and derivative:")
            print(r)
            print(p)
            print(derivative)
            break
        if np.all((abs(derivative) < threshold) + stop):
            r_not_found = False
            break
        stop[abs(derivative) < threshold] = True
        r, delta, stop = update_r(copy.deepcopy(r), derivative, delta, stop)
        counter += 1
        #if counter % 10 == 0:
        #    print("%i iterations" % counter)
        #    print(derivative)
        #    print(r)
        if counter == 2000:
            # jesli idzie tak dlugo to pewnie i tak cos jest nie tak.
            # to niech juz beda te estymacje ktore sa.
            break
    #print("r estimated:")
    #print(r)
    #print("***")
    return r
