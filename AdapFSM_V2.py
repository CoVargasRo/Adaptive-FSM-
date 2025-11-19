from AdapFSM_Functions_V2 import*

def AdapFSM_V2(state , ensamble , shots_1):
    """
    Protocolo AdapFSM con MLE

    Parameters:
        state : unknwown pure state to be estimated
        ensamble : number of copies of the state available
        shots_1: fraction of the ensemble used to measure the + and - elements of the FSM
        shots_2: fraction of the ensemble used to measure the adapted FSM 

    Return:
        Fidelities after first estimation and final estimation

    """

    shots_2 = 1 - shots_1

    d = len(state)
    
    #PASO 0: medir base computacional y definir estado fiducial
    BaseComp = np.eye(d, dtype=complex)
    meas = SimMeas(state, BaseComp, shots=1)

    #Elegir estado fiducial como el que tuvo mayor probabilidad
    b0 = np.max(meas)
    fid = np.where(meas == b0, meas, 0)/b0
    index = np.where(meas == b0)[0][0]
    # cambio de base, para que el fiducial sea el estado cero
    Base = BaseComp
    Base[:,0] = fid
    Base, R = np.abs(np.linalg.qr( Base )) #Gram-Schidth

    #PASO 1: construir FSM, medirlos, encontrar coeficientes y fases, primera estimación
    fsm_plus = FSM( d , -1 )
    fsm_plus_1 = Base@fsm_plus
    measures_accumulated = fsm_plus_1

    fsm_minus = FSM( d )
    fsm_minus_1 = Base@fsm_minus
    measures_accumulated = np.concatenate([measures_accumulated, fsm_minus_1], axis=1)

    #Medidas simuladas
    probs_plus  = SimMeas(state, fsm_plus_1, shots_1*ensamble/2)
    probs_minus = SimMeas(state, fsm_minus_1, shots_1*ensamble/2)

    probs_accumulated = probs_plus
    probs_accumulated = np.concatenate([probs_accumulated, probs_minus], axis=0)

    #Coeficientes de los elementos de medición
    b0 = fsm_plus[0]
    bk = fsm_plus[1:].real
    ck = fsm_plus[1:].imag

    #Cálculo del delta_k
    Delta_k = np.sum( ( (bk + 1j*ck)/b0 ) * ( probs_plus - probs_minus ), axis=1 )  

    #Ecuación para a0^2
    a0 = np.zeros(2*d-1)

    for i in range(2*d-1):
        num = np.abs( np.sum( (bk[:,i]-1j*ck[:,i]) * (Delta_k) ) )**2 - ( b0[i] )**2*np.sum(np.abs(Delta_k)**2) 
        den = 2*( probs_plus[i] + probs_minus[i]) - 4*( b0[i] )**2 

        a0[i] = np.real(np.sqrt( num / den ))

    mean_a0 = np.mean(a0) # choose the mean result of beta0

    # Estimación de los coef ak
    ak = Delta_k / (2*mean_a0)

    state_0 = np.hstack( ( mean_a0 , ak))
    state_hat = Base @ ( state_0 / np.linalg.norm( state_0 ) )

    #PASO 2: utilizar MLE para refinar la estimación

    state_hat_est = MLE_pure( d, probs_accumulated, measures_accumulated, state_hat ) #agregar datos anteriores
    Fid_1 = Fidelity(state_hat_est, state)
    
    #PASO 3: crear FSM con estimación anterior, medir y reconstruir con MLE

    Base = BaseComp
    Base[:,0] = state_hat_est
    Base, R = np.linalg.qr( Base ) #Gram-Schmidth
    fsm = Base @ fsm_minus
    measures_accumulated_4 = np.concatenate([measures_accumulated,fsm], axis=1)
    probs_3 = SimMeas( state, fsm , shots_2*ensamble)
    probs_accumulated_4 = np.concatenate([probs_accumulated,probs_3], axis =0)
    state_est = MLE_pure( d, probs_accumulated_4, measures_accumulated_4, state_hat_est ) #agregar datos anteriores

    Fid_2 = Fidelity(state_est, state)
    
    return Fid_1, Fid_2