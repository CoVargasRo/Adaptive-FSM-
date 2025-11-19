import numpy as np

# Construction of the Fisher Symmetric Measurements
def FSM( d, phase=1 ): #con phase=1, construye el povm negativo

    n = 2*d - 1 # number of measurement elements
    psi_k = np.zeros( (d, n), dtype=complex )

    psi_0 = np.ones(d, dtype=complex) 
    psi_0[1:] = - phase* np.exp( 1j*np.pi/4 )
    psi_0 = psi_0 / np.sqrt( n )

    psi_k[:,0] = psi_0

    for k in range(1,d):
        psi = np.ones(d, dtype=complex) 
        psi[1:] = - phase*np.exp( 1j*np.pi/4 ) / ( np.sqrt(n) + 1 )
        psi[k] += phase*np.sqrt(0.5*n)
        psi = psi / np.sqrt(n)

        psi_k[:,2*k-1] = psi 

        psi = np.ones(d, dtype=complex) 
        psi[1:] = - phase*np.exp( 1j*np.pi/4 ) / ( np.sqrt(n) + 1 )
        psi[k] += phase*1j*np.sqrt(0.5*n)
        psi = psi / np.sqrt(n)

        psi_k[:,2*k] = psi 

    return psi_k

#Elegir Estado Random
def RandomState( d ): 
    psi = np.random.randn( d ) + 1j*np.random.randn( d )
    psi = psi / np.linalg.norm( psi )
    return psi

# Medida Simulada
def SimMeas( state, measure, shots):

    probs = np.abs( measure.T.conj() @ state )**2

    result = np.random.multinomial(shots, probs , size=1)

    NewDistribution = result/shots

    return  NewDistribution[0]

# Fidelity between two pure states
def Fidelity( psi, phi ):  
    return np.abs(np.vdot( psi, phi ) )**2

# Reordering between complex and real vectors
def complex2real( A ):
    return np.concatenate( ( np.real(A), np.imag(A) ), axis=0 )    

def real2compelex( A ):
    d  = len(A)//2 
    return A[:d] + 1j*A[d:] 

#Maximum Likelihood Estimation for pure states
from scipy.optimize import minimize

def MLE_pure( d, probs_ex, 
                measures, 
                init_state=None ):  
    
    if init_state is None:
        init_state = RandomState( d )

    def likelihood( parametros ): 
        psi = real2compelex( parametros )
        psi = psi / np.linalg.norm( psi ) 
        probs_th = np.abs( measures.T.conj() @ psi ) ** 2
        return -np.sum(np.log10(probs_th) * probs_ex)
    
    beta_k = measures.real
    gamma_k = measures.imag

    def DerProbs(parametros):
        """"
        Derivatives of the probabilities with respect to the real and imaginary parameters of the state.
        """

        d = len(parametros)//2
        a = parametros[:d]
        b = parametros[d:]

        # normalizar el estado
        N = np.sqrt(np.sum(a**2 + b**2))
        a_tilde = a / N
        b_tilde = b / N

        # Probabilidades teoricas
        meas  = beta_k - 1j*gamma_k
        state = a + 1j*b
        state = state / N
        probs_th = np.abs( meas.T @ state)**2

        # derivadas parciales
        suma1_alpha = beta_k.T @ a_tilde   + gamma_k.T @ b_tilde   # hay n elementos en suma1, correspondientes a los alphas de las mediciones
        suma2_alpha = gamma_k.T @ a_tilde  - beta_k.T @ b_tilde  # hay n elementos en suma2

        der_ak = 2/N*(beta_k * suma1_alpha + gamma_k * suma2_alpha) - 2*a[:, np.newaxis] /N**2 * probs_th[np.newaxis, :]
        der_bk = 2/N*(gamma_k * suma1_alpha - beta_k * suma2_alpha) - 2*b[:, np.newaxis] /N**2 * probs_th[np.newaxis, :]

        return np.vstack((der_ak, der_bk)).T

    def MLE_jac(parametros):
        """"
        Gradient of the likelihood function.
        """
        # estado cuántico puro
        psi = real2compelex( parametros) 

        # probabilidades teoricas y experimentales
        probs_th = np.abs( measures.T.conj() @ psi)**2

        der = DerProbs(parametros)

        return -np.sum((probs_ex / probs_th) * der.T , axis = 1)

    results = minimize( likelihood, complex2real(init_state) , method='BFGS' , jac = MLE_jac)

    state_est = real2compelex( results.x )
    state_est = state_est / np.linalg.norm( state_est )
    
    return state_est