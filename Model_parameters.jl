# model_parameters.jl

module Para 

using LinearAlgebra,SparseArrays,PreallocationTools,Statistics
# Global parameters
const T_ref = 298.15  # Refernce Temperature[K]
const F = 96485.3329  # Faraday constant [C/mol]
const R = 8.314       # Ideal gas constant [J/(mol K)]



# Electrode parameters
const L_pos = 75.6e-6  # Positive electrode thickness [m]
const L_neg = 85.2e-6  # Negative electrode thickness [m]
const L_sep = 12.0e-6  # Separator thickness [m]
const R_pos = 5.22e-6  # Positive particle radius [m]
const R_neg = 5.86e-6  # Negative particle radius [m]
const A_electrode = 0.1 # Area of electrode [m^2]

const ϵ_pos_BOL = 0.335  # Positive electrode porosity
const ϵ_neg_BOL = 0.25  # Negative electrode porosity
const ϵ_sep_BOL = 0.47   # Separator porosity
const Brug_pos = ϵ_pos_BOL ^1.5  # Bruggeman coefficient in the positive electrode
const Brug_sep = ϵ_sep_BOL ^1.5  # Bruggeman coefficient

const a_neg_BOL = 3.84e5 # Negative particle surface area density [m^2/m^3]
const a_pos_BOL = 3.82e5 # Positive particle surface area density [m^2/m^3]
const ϵ_act_neg_BOL = 0.75  # Active material volume fraction in the negative electrode
const ϵ_act_pos_BOL = 0.665  # Active material volume fraction in the positive electrode

const N_FVM_pos_elec = 15  # Number of FVM elements in the positive electrode
const N_FVM_neg_elec = 15  # Number of FVM elements  in the negative electrode
const N_FVM_sep = 6  # Number of FVM elements in the separator
const N_FVM_electrode = N_FVM_pos_elec + N_FVM_neg_elec+ N_FVM_sep # Total number of FVM elements in the electrode
const N_FVM_particle =10 # Number of FVM elements in the particle (anod eand cathode)
const AbyV_outer_cntrl_vol_neg = 1.889097389e6 # Area to volume ratio of the last control volume in the negative particle
const AbyV_outer_cntrl_vol_pos = 2.120710862e6 # Area to volume ratio of the last control volume in the positive particle

const c_neg_max  =33133.0   # Maximum concentration of lithium in the negative particle [mol/m^3]
const c_pos_max  =  63104.0 # Maximum concentration of lithium in the positive particle [mol/m^3]
const c_e_ref    = 1000.0   # Reference concentration of lithium in the electrolyte [mol/m^3]

const t⁺ = 0.2594 # Transference number of lithium ions in the electrolyte
σ_neg = 0.18 # Conductivity of the negative electrode [S/m]
σ_pos = 215 # Conductivity of the positive electrode [S/m]

SOC_ini = 1.0         # Initial state of charge
θn_SOC0,θn_SOC100 = 0.02637,0.910612
θp_SOC0,θp_SOC100 = 0.8539736,0.263848
c_n_ini = c_neg_max *(θn_SOC0+(θn_SOC100-θn_SOC0)*SOC_ini)    #mol/m^3
c_p_ini = c_pos_max*(θp_SOC0+(θp_SOC100-θp_SOC0)*SOC_ini)   #mol/m^3
L_SEI_ini = 1e-9 # Initial thickness of SEI layer [m]

const R_SEI = 6.1e4 # SEI resistance [Ohm m^2]
const c0_SEI = 4541.0 # solvent concentration [mol/m^3]
const Molar_weight_SEI = 0.162 # Molar weight of solvent in SEI [kg/mol]
const rho_SEI = 1690 # Density of SEI [kg/m^3]



#Positive particle state matrix (Tridiagonal sparse matrix)(D_p=4e-15 m^2/s , 10 FVM elements, R_p=5.22e-6 m )
PosParticle_LDiag = [0.006291326148638871, 0.009271428008520441, 0.010712258036871591, 0.011551287354877928, 0.012098704131997828, 0.012483576294936972, 0.012768786207000784, 0.012988544306867347, 0.013163032938296088]
PosParticle_Diag=[-0.044039283040472096, -0.03145663074319435, -0.030132141027691434, -0.02975627232464331, -0.02960017384687469, -0.029520838082074702, -0.029475110696378964, -0.029446384518185483, -0.029427170695246332, -0.013163032938296088]   
PosParticle_UDiag= [0.044039283040472096, 0.025165304594555484, 0.020860713019170994, 0.019044014287771718, 0.018048886491996763, 0.01742213395007687, 0.01699153440144199, 0.016677598311184698, 0.016438626388378987]
const PosParticle_StateMatrix=Tridiagonal(PosParticle_LDiag,PosParticle_Diag,PosParticle_UDiag)

#Negative particle state matrix (Tridiagonal sparse matrix)(D_n=3.3e-14 m^2/s , 10 FVM elements, R_p=5.86e-6 m )
const NegParticle_StateMatrix = ( 6.546357558037951) .* PosParticle_StateMatrix


dx_neg = L_neg/N_FVM_neg_elec
dx_pos = L_pos/N_FVM_pos_elec
dx_sep = L_sep/N_FVM_sep
dx_neg_sep = (dx_neg + dx_sep)/2
dx_pos_sep = (dx_pos + dx_sep)/2

e_vec = 1 ./[dx_neg.*ones(N_FVM_neg_elec-1);dx_neg_sep;dx_sep.*ones(N_FVM_sep-1);dx_pos_sep;dx_pos.*ones(N_FVM_pos_elec-1)]
const Electrolyte_conc_interfaceGrad=spdiagm(N_FVM_electrode-1,N_FVM_electrode,0=>-e_vec,1=>e_vec)

left_weight = 0.5*ones(N_FVM_electrode-1)
right_weight = 0.5*ones(N_FVM_electrode-1)
left_weight[N_FVM_neg_elec],right_weight[N_FVM_neg_elec] = dx_neg/(dx_neg+dx_sep),dx_sep/(dx_neg+dx_sep)
left_weight[N_FVM_neg_elec+N_FVM_sep],right_weight[N_FVM_neg_elec+N_FVM_sep] = dx_sep/(dx_sep+dx_pos),dx_pos/(dx_sep+dx_pos)
const Mean_of_node_at_interface = spdiagm(N_FVM_electrode-1,N_FVM_electrode,0=>left_weight,1=>right_weight)

left_weight1 = 1 ./[dx_neg.*ones(N_FVM_neg_elec-1);dx_sep*ones(N_FVM_sep);dx_pos*ones(N_FVM_pos_elec)]
right_weight1 = 1 ./[dx_neg*ones(N_FVM_neg_elec);dx_sep*ones(N_FVM_sep);dx_pos*ones(N_FVM_pos_elec-1)]
const Electrolyte_divergence_matrix = spdiagm(N_FVM_electrode,N_FVM_electrode-1,0=>right_weight1,-1=>-left_weight1)


Source_neg = (1-t⁺)/(F*L_neg*A_electrode)
Source_sep = 0.0
Source_pos = -(1-t⁺)/(F*A_electrode*L_pos)
const electrolyte_source = [Source_neg*ones(N_FVM_neg_elec);Source_sep*ones(N_FVM_sep);Source_pos*ones(N_FVM_pos_elec)]




function Positive_OCP(surf_conc)
        
    sto=surf_conc/c_pos_max

    OCP=(
        -0.8090 * sto
        + 4.4875
        - 0.0428 * tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * tanh(15.9308 * (sto - 0.3120))
    )
    return OCP
end 


function Negative_OCP(surf_conc)

    sto=surf_conc/c_neg_max

    OCP = (
        1.9793 * exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 *tanh(29.8538 * (sto - 0.1234))
        - 0.04478 *tanh(14.9159 * (sto - 0.2769))
        - 0.0205 *tanh(30.4444 * (sto - 0.6103))
        )
    return OCP
end 

function Negative_exchange_current_density(c_e, c_s_surf, c_s_max, T)

    #T=298.15
    m_ref = 2.1e-5 #6.48e-7 # # (A/m2)(m3/mol)^1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = exp(E_r / 8.314 * (1 / 298.15 - 1 ./ T))
   
    if c_s_surf<0 || (c_s_max - c_s_surf) <0
        return  -(m_ref * arrhenius * c_e^0.5 * abs(c_s_surf)^0.5 * abs((c_s_max - c_s_surf)) ^ 0.5)
    end

    return  m_ref * arrhenius * c_e^0.5 * c_s_surf^0.5 * (c_s_max - c_s_surf) ^ 0.5
end

function Positive_exchange_current_density(c_e, c_s_surf, c_s_max, T)

    m_ref = 3.42e-6  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 17800
    arrhenius = exp(E_r / 8.314 * (1 / 298.15 - 1 ./ T))

    if c_s_surf<0 || (c_s_max - c_s_surf) <0
        return  -(m_ref * arrhenius * c_e^0.5 * abs(c_s_surf)^0.5 * abs((c_s_max - c_s_surf)) ^ 0.5)
    end
    return m_ref * arrhenius * c_e^0.5 * c_s_surf^0.5 * (c_s_max - c_s_surf)^0.5
    
end 


function electrolyte_conductivity(ce)

    return 1.297e-10*ce^3 - 7.937e-5*ce^1.5 + 3.329*ce 
end 

end 