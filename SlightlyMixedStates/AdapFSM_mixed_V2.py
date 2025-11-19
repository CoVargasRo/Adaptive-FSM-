import numpy as np

# Construction of the Fisher Symmetric Measurements
def FSM( d, phase=1 ): #with phase=1, the negative povm is constructed

    n = 2*d - 1
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

# Random pure state
def RandomState( d ): 
    psi = np.random.randn( d ) + 1j*np.random.randn( d )
    psi = psi / np.linalg.norm( psi )
    return psi

# Exact measurements for mixed state
def Meas_th(rho , measure):

    n = len(measure[0])  # amount of measure elements

    probs = np.zeros(n)

    for i in range(n):
        probs[i] = np.abs(measure[:,i].conj().T @ rho @ measure[:,i])

    return probs

# Simulated measurements for mixed state
def SimMeas( rho , measure, shots):

    n = len(measure[0])  # amount of measure elements

    shot = int(shots)

    probs = np.zeros(n)

    for i in range(n):
        probs[i] = np.abs(measure[:,i].conj().T @ rho @ measure[:,i])

    probs = probs / np.sum(probs)

    result = np.random.multinomial(shot, probs , size=1)

    NewDistribution = result/shot

    return  NewDistribution[0]

# Fidelity between pure states
def FidelityPure( psi, phi ):  
    return np.abs(np.vdot( psi, phi ) )**2

#############################################################################################
def complex2real( A ):
    return np.concatenate( ( np.real(A), np.imag(A) ), axis=0 )    

def real2compelex( A ):
    d  = len(A)//2 
    return A[:d] + 1j*A[d:] 


#Maximum Likelihood Estimation for sligthly mixed states
from scipy.optimize import minimize

def MLE_mixed( d , probs_exp , measures, pure_state = None , lamda = None ):

    if pure_state is None:
        pure_state = RandomState(d)

    if lamda is None:
        lamda = np.abs(np.random.normal(0, 0.001))

    # lambda parameterized with sigma function

    t = np.log(lamda/(1-lamda))

    pure_state_parameters = complex2real(pure_state)
    init_parameters = np.hstack((pure_state_parameters, t))

    def sigma(t):
        return 1/(1+np.exp(-t))

    # coefficients of the measurement elements FS
    beta_k = measures.real
    gamma_k = measures.imag

    varphi = np.sum(beta_k**2 + gamma_k**2, axis=0).real

    # Likelihood function

    def likelihood( parametros ): 
            psi = real2compelex( parametros[:-1] )
            psi = psi / np.linalg.norm( psi ) 
            t = parametros[-1]
    
            probs_th = (1-sigma(t))*np.abs(measures.T.conj()@psi)**2 + (sigma(t)/d)*varphi
            
            return -np.sum(np.log(probs_th) * probs_exp)
    
    beta_k = measures.real
    gamma_k = measures.imag

    def DerProbs(parametros):
        """"
        Derivatives of probabilities with respect to real and imaginary parameters of the state.
        """

        d = len(parametros)//2
        a = parametros[:d]
        b = parametros[d:-1]
        t = parametros[-1]

        lamda = sigma(t)

        # normalize the state
        N = np.sqrt(np.sum(a**2 + b**2))
        a_tilde = a / N
        b_tilde = b / N

        # theorical probabilities
        meas  = beta_k - 1j*gamma_k
        state = a + 1j*b
        state = state / N
        probs_th = np.abs( meas.T @ state)**2

        # partial derivatives
        suma1_alpha = beta_k.T @ a_tilde   + gamma_k.T @ b_tilde  
        suma2_alpha = gamma_k.T @ a_tilde  - beta_k.T @ b_tilde 

        der_ak = 2/N*(beta_k * suma1_alpha + gamma_k * suma2_alpha) - 2*a[:, np.newaxis] /N**2 * probs_th[np.newaxis, :]
        der_bk = 2/N*(gamma_k * suma1_alpha - beta_k * suma2_alpha) - 2*b[:, np.newaxis] /N**2 * probs_th[np.newaxis, :]

        return (1-lamda)*np.vstack((der_ak, der_bk)).T
    
    def Derivada_lamda(parametros):
        # Calculate the derivative of the probability of each measurement element with respect to the parameter a_k
        
        t = parametros[-1]
        lamda = sigma(t)
        d  = len(parametros)//2 
        ak = parametros[:d] 
        bk = parametros[d:-1] 

        sum1 = np.sum((beta_k - 1j*gamma_k).T * (ak + 1j*bk), axis=1)

        return lamda*(1-lamda)*(-np.abs(sum1)**2 + varphi/d)

    def MLE_jac(parametros):
        """"
        Likelihood function gradient.
        """
        state = real2compelex(parametros[:-1])
        psi = state / np.linalg.norm(state)
        t = parametros[-1]

        probs_th = (1-sigma(t))*np.abs(measures.T.conj()@psi)**2 + (sigma(t)/d)*varphi

        d_ab = DerProbs(parametros).T
        d_lamda = Derivada_lamda(parametros)

        der = np.vstack([d_ab , d_lamda])

        return -np.sum((probs_exp / probs_th) * der , axis = 1)

  
    ## Likelihood minimization

    results = minimize( likelihood , init_parameters , method='BFGS' , jac= MLE_jac)

    pure_state = real2compelex(results.x[:-1])
    state_est = pure_state / np.linalg.norm(pure_state)
    lamda = sigma(results.x[-1]).real

    return state_est , lamda


## AdapFSM protocol for mixed states

def AdapFSM_mixed_V2(psi , lamda ,ensamble, shots_1, shots_2):
    shots_3 = 1 - shots_1 - shots_2
    
    d = int(len(psi))

    rho = (1-lamda)*np.outer(psi, np.conj(psi)) + (lamda/d)*np.eye(d, dtype=complex)

    #STEP 0: Measure computational basis and define fiducial state
    BaseComp = np.eye(d, dtype=complex)
    #measures_accumulated = BaseComp
    meas = SimMeas(rho, BaseComp, shots_1*ensamble)
    #probs_accumulated = meas

    #Choosing fiducial state as the one with the highest probability
    max = np.max(meas)
    fid = np.where(meas == max, meas, 0)/max

    # change of basis, so that the fiducial is the zero state
    Base = BaseComp
    Base[:,0] = fid
    Base, R = np.abs(np.linalg.qr( Base )) #Gram-Schidth
    
    # change of base for measured probabilities
    meas = Base @ meas
    
    #####################################################################################
    #STEP 1: Build FSMs, measure them, find coefficients and phases, first estimate
    fsm_plus = FSM( d , -1 )
    fsm_plus_1 = Base@fsm_plus
    measures_accumulated = fsm_plus_1

    fsm_minus = FSM( d )
    fsm_minus_1 = Base@fsm_minus
    measures_accumulated = np.concatenate([measures_accumulated, fsm_minus_1], axis=1)

    #Simulated measurements
    probs_plus  = SimMeas(rho, fsm_plus_1 , shots_2*ensamble/2)
    probs_minus = SimMeas(rho, fsm_minus_1, shots_2*ensamble/2)

    probs_accumulated = probs_plus
    probs_accumulated = np.concatenate([probs_accumulated, probs_minus], axis=0)

    #Coefficients of the measuring elements
    b0 = fsm_plus[0]
    bk = fsm_plus[1:].real
    ck = fsm_plus[1:].imag

    varphi = (b0**2 + np.sum( bk**2 + ck**2 , axis=0)).real

    #Calculation of delta_k
    Delta_k = np.sum( ( (bk + 1j*ck)/b0 ) * ( probs_plus - probs_minus ), axis=1 )  

    #Calculation of lambda
    lamda_k = np.abs((d/2) * ((meas[0] + meas[1:] - np.sqrt((meas[0] - meas[1:])**2 + np.abs(Delta_k**2)))))

    lamda = np.mean(lamda_k)

    #Equation for a0^2
    a0 = np.zeros(2*d-1)

    for i in range(2*d-1):
        num = np.abs( np.sum( (bk[:,i]-1j*ck[:,i]) * (Delta_k) ) )**2 - (1 - lamda )*( b0[i] )**2*np.sum(np.abs(Delta_k)**2) 
        den = 2*(1 - lamda)*( probs_plus[i] + probs_minus[i] - (lamda/d) * varphi[i] ) - 4*(1 - lamda)**2*( b0[i] )**2 

        a0[i] = np.real(np.sqrt( num / den ))

    mean_a0 = np.mean(a0) # choose the mean result of beta0

    # Estimation of the ak coefficients
    ak = Delta_k / (2*(1-lamda)*mean_a0)

    state_0 = np.hstack( ( mean_a0 , ak))
    state_hat = Base @ ( state_0 / np.linalg.norm( state_0 ) )

    #Fid0 = FidelityPure(psi , state_hat)  #Fidelidad entre el estado estimado y el real

    #STEP 2: Use MLE to refine the estimate

    state_hat_est , lamda = MLE_mixed( d , probs_accumulated, measures_accumulated, state_hat , lamda ) #add the previous data

    Fid1 = FidelityPure(state_hat_est, psi)  #Accuracy between the estimated and actual state

    #STEP 3: Create FSM with previous estimate, measure and estimate with MLE

    Base = BaseComp
    Base[:,0] = state_hat_est
    Base, R = np.linalg.qr( Base ) #Gram-Schmidth
    fsm = Base @ fsm_minus
    measures_accumulated_4 = np.concatenate([measures_accumulated,fsm], axis=1)

    ## Simulated measurements
    probs_3 = SimMeas( rho, fsm, shots_3*ensamble)
    probs_accumulated_4 = np.concatenate([probs_accumulated,probs_3], axis =0)
    psi_est , lamda = MLE_mixed( d , probs_accumulated_4 , measures_accumulated_4 , state_hat_est , lamda ) #add previous data

    Fid2 = FidelityPure(psi_est, psi)

    return Fid1 , Fid2