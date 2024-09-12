import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc


data = np.loadtxt("196keV.Xe")
X = data[:, 0] #channels
Y = data[:, 1] #counts

counts, bin_edges = np.histogram(X, bins=len(X), weights=Y) 
bin_centers = X
counts_err = np.sqrt(counts)


def gaussiana(x, mu, A, sig, A_bcg, B_bcg, lmdR1, muR1, sigR11, sigR12, lmdR2, muR2, sigR21, sigR22):
    return(A * np.exp(-((x - mu)**2) / (2 * sig**2)) )

def fondo(x, mu, A, sig, A_bcg, B_bcg, lmdR1, muR1, sigR11, sigR12, lmdR2, muR2, sigR21, sigR22):
    return( A_bcg * erfc((x - mu) / sig) + B_bcg)

def ajuste1(x, mu, A, sig, A_bcg, B_bcg, lmdR1, muR1, sigR11, sigR12, lmdR2, muR2, sigR21, sigR22):
    return( lmdR1 * np.exp(-(x - muR1) / sigR11) * erfc(-(x - muR1) / sigR12) )

def ajuste2(x, mu, A, sig, A_bcg, B_bcg, lmdR1, muR1, sigR11, sigR12, lmdR2, muR2, sigR21, sigR22):
    return( lmdR2 * np.exp((x - muR2) / sigR21) * erfc((x - muR2) / sigR22) )

def ajuste_total(x, mu, A, sig, A_bcg, B_bcg, lmdR1, muR1, sigR11, sigR12, lmdR2, muR2, sigR21, sigR22):
    return( gaussiana(x, mu, A, sig, A_bcg, B_bcg, lmdR1, muR1, sigR11, sigR12, lmdR2, muR2, sigR21, sigR22) + fondo(x, mu, A, sig, A_bcg, B_bcg, lmdR1, muR1, sigR11, sigR12, lmdR2, muR2, sigR21, sigR22) + ajuste1(x, mu, A, sig, A_bcg, B_bcg, lmdR1, muR1, sigR11, sigR12, lmdR2, muR2, sigR21, sigR22)+ajuste2(x, mu, A, sig, A_bcg, B_bcg, lmdR1, muR1, sigR11, sigR12, lmdR2, muR2, sigR21, sigR22))


#def Gauss1slb1RT1LT(x, mu, A, sig, A_bcg, B_bcg, lmdR1, muR1, sigR11, sigR12, lmdR2, muR2, sigR21, sigR22):
#    return (A * np.exp(-((x - mu)**2) / (2 * sig**2)) + A_bcg*erfc((x - mu) / sig) + B_bcg + lmdR1*np.exp(-(x - muR1) / sigR11) * erfc(-(x - muR1) / sigR12) + lmdR2 * np.exp((x - muR2) / sigR21) * erfc((x - muR2) / sigR22))



#GAUSSIANA
mu_initial = 1.14101070e+03
A_initial = 6.10527425e+05 #amplitud de la gaussiana
sig_initial = 2.75765864e+00

#FONDO
A_bcg_initial = 1500 #-4.09754145e+03
B_bcg_initial = 2700 #1e+00 #-8.62940265e+06 #cte traslación y

#AJUSTE1
lmdR1_initial = 20000 #6e+03 #amplitud
muR1_initial = 1145 #1.14101070e+03 #centro fn error
sigR11_initial = 3.5 #2.11996922e+05#? pero lo hizo plano
sigR12_initial = 2 #2.11996922e+01 #estira 

#AJUSTE2
lmdR2_initial = 5000 #3e+03 #amplitud
muR2_initial = 1135 #1.14101070e+03 #centro fn error 
sigR21_initial = 2.5  #2.07223347e+04 
sigR22_initial = 2 #2.07223347e+01 #estira #cambié a 1 y se mejoró, puse 0 y izq murio

initial_params = [mu_initial, A_initial, sig_initial, A_bcg_initial, B_bcg_initial, lmdR1_initial, muR1_initial, sigR11_initial, sigR12_initial, lmdR2_initial, muR2_initial, sigR21_initial, sigR22_initial]

popt, pcov = curve_fit(ajuste_total, X, counts, p0=initial_params, maxfev=100000)


residuals = counts - ajuste_total(X, *popt)
ss_res = np.sum((residuals/counts_err)**2)
chi_squared_reduced = ss_res / (len(counts) - len(popt))
print('Chi cuadrado reducido =', chi_squared_reduced)
print(popt)
print(f"incertidumbres: dmu={round(np.sqrt(pcov[0, 0]), 4)}, dsigma={round(np.sqrt(pcov[2, 2]), 4)}")


#calculation of the TOTAL NUMBER OF COUNTS OF THIS PEAK
total_counts = np.sum(counts) #también serviría np.sum(ajuste_total(X, *popt))
background_counts = np.sum(fondo(X, *popt))
peak_counts = total_counts - background_counts

print(f"Número de cuentas = {round(peak_counts , 4)}" )
Energia = (4.016979010535439e-08)*(popt[0]**2)+ 0.17159686335011792*popt[0] + 0.48508270091086614 
print(f"La energía es {Energia}keV")

"""
# Gráfico de residuos
plt.figure()
plt.scatter(X, residuals, color='blue')
plt.hlines(0, min(X), max(X), colors='red', linestyles='dashed')
plt.xlabel('Canales')
plt.ylabel('Residuales')
plt.title('Análisis de residuos')
plt.grid(True)
plt.show()
"""


plt.figure(figsize=(4,2))
plt.bar(X, counts, align="center", alpha=0.7, color="grey")

plt.plot(X, gaussiana(X, *popt), color='green', label='Gaussian')

plt.plot(X, fondo(X, *popt), color='blue', label='erfc+C')

plt.plot(X, ajuste1(X, *popt), color='red', label='exp*erfc')

plt.plot(X, ajuste2(X, *popt), color='orange', label='exp*erfc')

#plt.plot(X, ajuste_total(X, mu_initial, A_initial, sig_initial, A_bcg_initial, B_bcg_initial, lmdR1_initial, muR1_initial, sigR11_initial, sigR12_initial, lmdR2_initial, muR2_initial, sigR21_initial, sigR22_initial), color='red', label='Total')

plt.plot(X, ajuste_total(X, *popt), color='black', label='Fit')

#plt.xlim(1098,1175)
plt.ylim(200,750000)
plt.xlabel('Channels')
plt.ylabel('Counts')
plt.yscale("log")
plt.title('Fitting of the 196keV peak')
plt.legend()
plt.show()